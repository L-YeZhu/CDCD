import os

string = "python train.py --name coco_train --config_file configs/coco.yaml --num_node 1 --tensorboard --load_path /home/zhuye/VQ-Diffusion_d2m/OUTPUT/pretrained_model/CC_pretrained.pth --output /data/zhuye/Text2Image"

os.system(string)

