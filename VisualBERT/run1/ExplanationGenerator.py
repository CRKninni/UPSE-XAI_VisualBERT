import numpy as np
import torch
import cv2
import copy
from scipy.sparse.linalg import eigs
import torch.nn.functional as F
import math
from pymatting.util.util import row_sum
from scipy.sparse import diags
from sklearn.decomposition import PCA

def get_grad_cam(grads,cams,modality):
    final_gradcam = []
    num_layers = len(cams)
    for i in range(num_layers):
        grad = grads[i]  
        cam = cams[i]

        layer_gradcam = cam * grad 
    
        layer_gradcam = layer_gradcam.clamp(min=0).mean(dim=0)

        final_gradcam.append(layer_gradcam.cpu().detach().numpy())
        

    heatmap = np.mean(final_gradcam, axis=0)
    heatmap = np.mean(heatmap, axis=0)
    heatmap = heatmap.flatten()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)    
    heatmap = torch.tensor(heatmap, dtype=torch.float32)

    return heatmap


def get_rollout(cams, modality):
    num_layers = len(cams)
    x = None

    for i in range(num_layers):
        cam_i = cams[i]
            
        cam_i_avg = cam_i.mean(dim=0) 
        
        if x is None:
            x = cam_i_avg.clone()
        else:
            x = x * cam_i_avg  # Element-wise multiplication
            x = x / torch.norm(x, p=2)
            
    gradcam = x.detach().cpu().numpy()

    heatmap = np.mean(gradcam, axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = heatmap.flatten()
    heatmap = torch.tensor(heatmap, dtype=torch.float32)
    
    return heatmap

def eig_seed(feats, modality, iters):
    patch_scores_norm = get_eigs(feats, modality, how_many = 5)
    temp = patch_scores_norm[:12]
    patch_scores_norm = patch_scores_norm[-100:]
    num_patches = 10
    heatmap = patch_scores_norm.reshape(num_patches, num_patches)  # Shape: [num_patches, num_patches]

    
    seed_index = np.argmax(patch_scores_norm)

    # Convert the 1D index to 2D indices
    seed_row = seed_index // num_patches
    seed_col = seed_index % num_patches


    # Initialize a mask for the expanded seed region
    seed_mask = np.zeros_like(heatmap)
    seed_mask[seed_row, seed_col] = 1

    # Define the number of expansion iterations
    num_expansion_iters = iters

    # Perform seed expansion
    for _ in range(num_expansion_iters):
        # Find neighboring patches
        neighbor_mask = cv2.dilate(seed_mask, np.ones((3, 3), np.uint8), iterations=1)
        neighbor_mask = neighbor_mask - seed_mask  # Exclude already included patches
        neighbor_indices = np.where(neighbor_mask > 0)
        
        # For each neighbor, decide whether to include it based on similarity
        for r, c in zip(*neighbor_indices):
            # Use heatmap values as similarity scores
            similarity = heatmap[r, c]
            # Define a threshold for inclusion
            threshold = 0.5  # Adjust this value as needed
            
            if similarity >= threshold:
                seed_mask[r, c] = 1  # Include the neighbor
            else:
                seed_mask[r, c] = 0.001

    # Apply the seed mask to the heatmap
    refined_heatmap = heatmap * seed_mask
    refined_heatmap = refined_heatmap.flatten()
    refined_heatmap = np.concatenate((temp, refined_heatmap))
    refined_heatmap = (refined_heatmap - refined_heatmap.min()) / (refined_heatmap.max() - refined_heatmap.min() + 1e-8)
    refined_heatmap = torch.tensor(refined_heatmap, dtype=torch.float32)
    
    return refined_heatmap
    
def get_pca_component(feats, modality, component=0, device="cpu"):
    if feats.size(0) == 1:
        feats = feats.detach().squeeze()

    # if modality == "image":
    #     n_image_feats = feats.size(0)
    #     val = int(math.sqrt(n_image_feats))
    #     if val * val == n_image_feats:
    #         feats = F.normalize(feats, p=2, dim=-1).to(device)
    #     elif val * val + 1 == n_image_feats:
    #         feats = F.normalize(feats, p=2, dim=-1)[1:].to(device)
    #     else:
    #         print(f"Invalid number of features detected: {n_image_feats}")
    # else:
    #     feats = F.normalize(feats, p=2, dim=-1)[1:-1].to(device)


    feats_reshaped = feats.cpu().detach().numpy()

    pca = PCA(n_components=5)
    principal_components = pca.fit_transform(feats_reshaped)

    second_pc = principal_components[:, component]

    second_pc = torch.tensor(second_pc, dtype=torch.float32).to(device)


    second_pc = torch.abs(second_pc)
    
    second_pc_norm = (second_pc - second_pc.min()) / (second_pc.max() - second_pc.min() + 1e-8)
    
    return second_pc_norm


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    matrices_aug = all_layer_matrices
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

def get_diagonal (W):
    D = row_sum(W)
    D[D < 1e-12] = 1.0  # Prevent division by zero.
    D = diags(D)
    return D

def get_eigs (feats, modality, how_many = None):
    if feats.size(0) == 1:
        feats = feats.detach().squeeze()


    if modality == "image":
        feats = F.normalize(feats, p = 2, dim = -1)

    else:
        feats = F.normalize(feats, p = 2, dim = -1)[1:-1]
        # feats = feats[1:-1]


    W_feat = (feats @ feats.T)
    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max() 

    W_feat = W_feat.detach().cpu().numpy()

    
    D = np.array(get_diagonal(W_feat).todense())

    L = D - W_feat

    L_shape = L.shape[0]
    if how_many >= L_shape - 1: 
        how_many = L_shape - 2

    try:
        eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5, M = D)
    except:
        try:
            eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5)
        except:
            eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM')
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
    
    n_tuple = torch.kthvalue(eigenvalues.real, 2)
    fev_idx = n_tuple.indices
    fev = eigenvectors[fev_idx]

    if modality == 'text':
        fev = torch.cat( ( torch.zeros(1), fev, torch.zeros(1)  ) )

    return fev


class SelfAttentionGenerator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_transformer_att(self, input, index=None, start_layer=0, save_visualization=False, save_visualization_per_token=False):
        output = self.model(input)['scores']
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        cams = []
        blocks = self.model.model.bert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = rollout[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='trans_att')

        if save_visualization_per_token:
            for token in range(1,cls_index+1):
                token_relevancies = rollout[:, token]
                token_relevancies[:, token] = 0
                save_visual_results(input, token_relevancies, method_name='trans_att', expl_path='per_token', suffix=str(token))
        return cls_per_token_score

    def generate_ours1(self, input, index=None, save_visualization=False, save_visualization_per_token=False):
        output = self.model(input)['scores']
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        try:
            one_hot = torch.sum(one_hot.cuda() * output)
        except:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        blocks = self.model.model.bert.encoder.layer
        num_tokens = blocks[0].attention.self.get_attn().shape[-1]
        R = torch.eye(num_tokens, num_tokens).to(blocks[0].attention.self.get_attn().device)
        grads = []
        cams = []
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            grads.append(grad)
            cams.append(cam)
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            R += torch.matmul(cam, R)
            
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = R[cls_index]
        cls_per_token_score[:, cls_index] = 0
        
        feats = blocks[-1].intermediate_outputs[0]
        dsm = get_eigs(feats,'image',how_many=2)
        lost = eig_seed(feats,'image',15)
        pca_0 = get_pca_component(feats,'image',0)
        pca_1 = get_pca_component(feats,'image',1)
        grad_cam = get_grad_cam(grads,cams,'image')
        rollout = get_rollout(cams,'image')
        grad_cam = grad_cam * 2
        pca_1 = pca_1 * 0.001
        dsm = dsm[:len(cls_per_token_score)]
        lost = lost[:len(cls_per_token_score)]
        pca_0 = pca_0[:len(cls_per_token_score)]
        pca_1 = pca_1[:len(cls_per_token_score)]
        
        
        
        x = dsm + lost + pca_0 + pca_1 + grad_cam + rollout
        x = (x-x.min())/(x.max()-x.min()+1e-8)
        x = x.unsqueeze(0)
        x = abs(x.to(blocks[0].attention.self.get_attn().device))
        x[:, cls_index] = 0
        
    
        cls_per_token_score = x
        
        # dsm=dsm[:len(cls_per_token_score)]
        # dsm = abs(dsm.to(blocks[0].attention.self.get_attn().device))
        # cls_per_token_score = cls_per_token_score + dsm
        

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='ours')

        if save_visualization_per_token:
            for token in range(1,cls_index+1):
                token_relevancies = R[:, token]
                token_relevancies[:, token] = 0
                save_visual_results(input, token_relevancies, method_name='ours', expl_path='per_token', suffix=str(token))
        return cls_per_token_score


    def generate_ours(self, input, index=None, save_visualization=False, save_visualization_per_token=False):
        output = self.model(input)['scores']
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        blocks = self.model.model.bert.encoder.layer
        num_tokens = blocks[0].attention.self.get_attn().shape[-1]
        R = torch.eye(num_tokens, num_tokens).to(blocks[0].attention.self.get_attn().device)
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            R += torch.matmul(cam, R)
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = R[cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='ours')

        if save_visualization_per_token:
            for token in range(1,cls_index+1):
                token_relevancies = R[:, token]
                token_relevancies[:, token] = 0
                save_visual_results(input, token_relevancies, method_name='ours', expl_path='per_token', suffix=str(token))
        return cls_per_token_score

    def generate_partial_lrp(self, input, index=None, save_visualization=False):
        output = self.model(input)['scores']
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1

        self.model.relprop(torch.tensor(one_hot).to(output.device), **kwargs)

        cam = self.model.model.bert.encoder.layer[-1].attention.self.get_attn_cam()[0]
        # cam = cam.clamp(min=0).mean(dim=0).unsqueeze(0)
        cam = cam.mean(dim=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = cam[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='partial_lrp')
        return cls_per_token_score

    # def generate_full_lrp(self, input_ids, attention_mask,
    #                  index=None):
    #     output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
    #     kwargs = {"alpha": 1}
    #
    #     if index == None:
    #         index = np.argmax(output.cpu().data.numpy(), axis=-1)
    #
    #     one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    #     one_hot[0, index] = 1
    #     one_hot_vector = one_hot
    #     one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    #     one_hot = torch.sum(one_hot.cuda() * output)
    #
    #     self.model.zero_grad()
    #     one_hot.backward(retain_graph=True)
    #
    #     cam = self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)
    #     cam = cam.sum(dim=2)
    #     cam[:, 0] = 0
    #     return cam

    def generate_raw_attn(self, input, save_visualization=False):
        output = self.model(input)['scores']
        cam = self.model.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
        cam = cam.mean(dim=0).unsqueeze(0)
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = cam[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='raw_attn')
        return cls_per_token_score

    def generate_rollout(self, input, start_layer=0, save_visualization=False):
        output = self.model(input)['scores']
        blocks = self.model.model.bert.encoder.layer
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attention.self.get_attn()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = rollout[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='rollout')
        return cls_per_token_score

    def generate_attn_gradcam(self, input, index=None, save_visualization=False):
        output = self.model(input)['scores']

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        cam = self.model.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
        grad = self.model.model.bert.encoder.layer[-1].attention.self.get_attn_gradients()[0]

        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = cam[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='gradcam')
        return cls_per_token_score

# class MultiAttentionGenerator:
#     def __init__(self, model):
#         self.model = model
#         self.model.eval()
#
#     def generate_LRP(self, input, index=None, start_layer=0, save_visualization=False, save_visualization_per_token=False):
#         output = self.model(input)['scores']
#         kwargs = {"alpha": 1}
#
#         if index == None:
#             index = np.argmax(output.cpu().data.numpy(), axis=-1)
#
#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0, index] = 1
#         one_hot_vector = one_hot
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         one_hot = torch.sum(one_hot.cuda() * output)
#
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)
#
#         # self.model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)
#
#         cams = []
#         blocks = self.model.model.bert.encoder.layer
#         for blk in blocks:
#             grad = blk.attention.self.get_attn_gradients()
#             cam = blk.attention.self.get_attn_cam()
#             cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
#             grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
#             cam = grad * cam
#             cam = cam.clamp(min=0).mean(dim=0)
#             cams.append(cam.unsqueeze(0))
#         rollout = compute_rollout_attention(cams, start_layer=start_layer)
#         input_mask = input['input_mask']
#         cls_index = input_mask.sum(1) - 2
#         cls_per_token_score = rollout[0, cls_index]
#         cls_per_token_score[:, cls_index] = 0
#
#         if save_visualization:
#             save_visual_results(input, cls_per_token_score, method_name='ours')
#
#         if save_visualization_per_token:
#             for token in range(1,cls_index+1):
#                 token_relevancies = rollout[:, token]
#                 token_relevancies[:, token] = 0
#                 save_visual_results(input, token_relevancies, method_name='ours', expl_path='per_token', suffix=str(token))
#         return cls_per_token_score
