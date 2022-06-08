from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from image_synthesis.utils.misc import instantiate_from_config
from tqdm import tqdm
import pickle
import glob

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class Ade20kDataset(Dataset):
    def __init__(self, data_root, negative_sample_path, phase = 'train', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.condition_folder = os.path.join(data_root,'annotations')
        self.image_folder = os.path.join(data_root,'images')
        self.phase = phase
        if self.phase == 'train':
            self.image_folder = os.path.join(self.image_folder, 'training')
            self.condition_folder = os.path.join(self.condition_folder, 'training')
            self.images = glob.glob(self.image_folder+'/*')
        else:
            self.image_folder = os.path.join(self.image_folder, 'validation')
            self.condition_folder = os.path.join(self.condition_folder, 'validation')
            self.images = glob.glob(self.image_folder+'/*')
        self.num = len(self.images)

        print("check images numbers:", phase, self.num)


    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        image_path = self.images[index]
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        print("check image path:", image_path, np.shape(image))
        image_name = image_path.split('/')[-1].replace('.jpg','.png')
        semantic_path = os.path.join(self.condition_folder, image_name)
        semantic = load_img(semantic_path)
        semantic = self.transform(image = semantic)['image']
        print("check semantic path:", semantic_path, np.shape(semantic))
        data = {
            'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
            'semantic': np.transpose(semantic.astype(np.float32), (2, 0, 1)),
        }
    
        return data


