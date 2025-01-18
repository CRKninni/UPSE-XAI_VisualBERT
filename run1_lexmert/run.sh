export PYTHONPATH=/home/crk/ExplanableAI/Transformer-MM-Explainability


nohup python /home/crk/ExplanableAI/Transformer-MM-Explainability/lxmert/lxmert/perturbation.py  --COCO_path /home/crk/val2014/ --method ours_no_lrp --is-positive-pert true > progress3.log &
wait
nohup python /home/crk/ExplanableAI/Transformer-MM-Explainability/lxmert/lxmert/perturbation.py  --COCO_path /home/crk/val2014/ --method ours_no_lrp > progress4.log &
wait
nohup python /home/crk/ExplanableAI/Transformer-MM-Explainability/lxmert/lxmert/perturbation.py  --COCO_path /home/crk/val2014/ --method ours_no_lrp --is-text-pert true  --is-positive-pert true > progress1.log &
wait
nohup python /home/crk/ExplanableAI/Transformer-MM-Explainability/lxmert/lxmert/perturbation.py  --COCO_path /home/crk/val2014/ --method ours_no_lrp --is-text-pert true > progress2.log &
