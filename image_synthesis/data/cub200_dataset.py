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

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class Cub200Dataset(Dataset):
    def __init__(self, data_root, negative_sample_path = None, phase = 'train', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.image_folder = os.path.join(data_root, 'images')
        self.root = os.path.join(data_root, phase)
        pickle_path = os.path.join(self.root, "filenames.pickle")
        self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")
        self.negative_sample_path = negative_sample_path
        self.num = len(self.name_list)
        if self.negative_sample_path != None:
            with open(negative_sample_path, 'r') as f:
                self.extra_img = json.load(f)

        # load all caption file to dict in memory
        self.caption_dict = {}


        # print("check name_list:", len(self.name_list))
        # exit()

        # for index in tqdm(range(self.num)):
        #     name = self.name_list[index]
        #     this_text_path = os.path.join(data_root, 'text', 'text', name+'.txt')
        #     image_path = os.path.join(self.image_folder, name+'.jpg')
        #     if not os.path.exists(image_path) or not os.path.exists(this_text_path):
        #         print("missing file:", image_path, this_text_path)


        
        for index in tqdm(range(self.num)):
            name = self.name_list[index]
            this_text_path = os.path.join(data_root, 'text', 'text', name+'.txt')
            # print("check name and text path:", index, name, this_text_path)
            with open(this_text_path, 'r') as f:
                caption = f.readlines()
            self.caption_dict[name] = caption

        print("load caption file done")


    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        name = self.name_list[index]
        image_path = os.path.join(self.image_folder, name+'.jpg')
        # if os.path.exists(image_path):
        #     print(index, image_path)
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        caption_list = self.caption_dict[name]
        caption = random.choice(caption_list).replace('\n', '').lower()
        # else:
        if self.negative_sample_path != None:
            neg_sample = self.extra_img[index]
            for i in range(len(neg_sample)):
                img = load_img(neg_sample[i])
                img = np.array(img).astype(np.uint8)
                img = self.transform(image = img)['image']
                if i == 0:
                    neg_img = np.expand_dims(img, axis=0)
                else:
                    img = np.expand_dims(img, axis=0)
                    neg_img = np.concatenate((neg_img, img), axis=0) 

        else:
            neg_img = None

        # print("check data loader:", np.shape(image), np.shape(neg_img))
        
        data = {
                'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                'text': caption,
                'negative_img': np.transpose(neg_img.astype(np.float32), (0, 3, 1, 2)),
        }
    
        return data


