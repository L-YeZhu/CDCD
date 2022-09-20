from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from synthesis.utils.misc import instantiate_from_config

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class CocoDataset(Dataset):
    def __init__(self, data_root, negative_sample_path ,phase = 'train', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.root = os.path.join(data_root, phase)
        # input_file = os.path.join(data_root, input_file)
        caption_file = "captions_"+phase+"2014.json"
        caption_file = os.path.join(data_root, "annotations", caption_file)

        self.json_file = json.load(open(caption_file, 'r'))
        print("length of the dataset is ")
        print(len(self.json_file['annotations']))

        # print("check json_file:", self.json_file['annotations'][1])
        # exit()

        self.num = len(self.json_file['annotations'])
        self.image_prename = "COCO_" + phase + "2014_"
        self.folder_path = os.path.join(data_root, phase+'2014', phase+'2014')


        self.negative_sample_path = negative_sample_path
        self.phase = phase
        if self.phase == 'train' and self.negative_sample_path != None:
            # print("negative_sample_path:", negative_sample_path)
            with open(negative_sample_path, 'r') as f:
                self.extra_img = json.load(f)
            # self.extra_img = os.path.join()
            print("negative_sample_path:", negative_sample_path, len(self.extra_img))
            # print("check path:", self.extra_img[0])
        else:
            self.extra_img = None


 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        this_item = self.json_file['annotations'][index]
        caption = this_item['caption'].lower()
        # print("check data loader:", this_item, caption)
        image_name = str(this_item['image_id']).zfill(12)
        image_path = os.path.join(self.folder_path, self.image_prename+image_name+'.jpg')
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        if self.phase == 'train' and self.extra_img != None:
            neg_sample = self.extra_img[index]
            for i in range(len(neg_sample)):
                neg_img_name = str(neg_sample[i]).zfill(12)
                neg_img_path = os.path.join(self.folder_path, self.image_prename+neg_img_name+'.jpg')
                # print("neg_img_path:", i, neg_img_path)
                img = load_img(neg_img_path)
                img = np.array(img).astype(np.uint8)
                img = self.transform(image = img)['image']
                if i == 0:
                    neg_img = np.expand_dims(img, axis=0)
                else:
                    img = np.expand_dims(img, axis=0)
                    neg_img = np.concatenate((neg_img, img), axis=0)                
            data = {
                    'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                    'text': caption,
                    'negative_img': np.transpose(neg_img.astype(np.float32), (0, 3, 1, 2)),
                }
        else:
            data = {
                    'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                    'text': caption,
                }            

        return data
