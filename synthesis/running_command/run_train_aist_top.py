import os

string = "python train.py --name aist_train_top_vanilla --config_file configs/aist_top.yaml --num_node 1 --tensorboard --load_path OUTPUT/pretrained_model/CC_pretrained.pth --output /data/zhuye/D2M-Diffusion"

os.system(string)