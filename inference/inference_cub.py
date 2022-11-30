# ------------------------------------------
# Modified based on VQ-Diffusion Project.
# Licensed under the MIT License.
# ------------------------------------------

import os
import sys
import pickle
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image

from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import get_model_parameters_info

class VQ_Diffusion():
    def __init__(self, config, path):
        self.info = self.get_model(ema=True, model_path=path, config_path=config)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad=False

    def get_model(self, ema, model_path, config_path):
        if 'OUTPUT' in model_path: # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else: 
            model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)
        model = build_model(config)
        model_parameters = get_model_parameters_info(model)
        
        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")

        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']
        elif 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 0

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)

        if ema==True and 'ema' in ckpt:
            print("Evaluate EMA model")
            ema_model = model.get_ema_model()
            missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)
        
        return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

    def inference_generate_sample_with_class(self, text, truncation_rate, save_root, batch_size,fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['label'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+'r',
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.jpg')
            im = Image.fromarray(content[b])
            im.save(save_path)

    def inference_generate_sample_with_condition(self, text, truncation_rate, save_root, batch_size, count,fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['text'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root, exist_ok=True)
        #os.makedirs(save_root_, exist_ok=True)

        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            #print("check str_cond:",str_cond)
            str_cond = str_cond.replace(" ","").replace(".","_").replace("/","")
            #print("check new str_cond:",str_cond)
            #exit()
            save_base_name = str_cond + save_base_name
            save_path = os.path.join(save_root, str(count)+'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)


if __name__ == '__main__':

    # Please specify your paths to different files here
    VQ_Diffusion = VQ_Diffusion(config='/configs/cub200.yaml', path='./checkpoint.pth')
    data_root = "/data/zhuye/CUB-200-2011/CUB_200_2011/"
    pickle_path = "/data/zhuye/CUB-200-2011/CUB_200_2011/test/filenames.pickle"
    name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")
    num = len(name_list)
    #caption_file = open("cub200_input_caption_b.txt", 'w')
    count = 0
    for index in tqdm(range(num)):
        name = name_list[index]
        this_text_path = os.path.join(data_root, 'text', 'text', name+'.txt')
        with open(this_text_path, 'r') as f:
            caption = f.readlines()
        for i, cap in enumerate(caption):
            count += 1
            print(index, i, cap)
            VQ_Diffusion.inference_generate_sample_with_condition(cap,truncation_rate=0.85, save_root="/data/zhuye/Text2Image/cub200_train_cd_sample_t100/syn_360",batch_size=1, count=count)

# print("total cap count:", count)





