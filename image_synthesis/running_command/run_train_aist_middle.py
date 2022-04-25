import os

string = "python train.py --name aist_train_middle_vanilla --config_file configs/aist_middle.yaml --num_node 1 --tensorboard --load_path OUTPUT/pretrained_model/CC_pretrained.pth --output /data/zhuye/D2M-Diffusion"

os.system(string)