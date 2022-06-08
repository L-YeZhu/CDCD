from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from image_synthesis.utils.misc import instantiate_from_config

def load_img(filepath):
    # print("check filepath:", filepath)
    # exit()
    img = Image.open(filepath).convert('RGB')
    return img

class ImageNetDataset(Dataset):
    def __init__(self, data_root, input_file, inter_negative_samples, phase = 'train', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.root = os.path.join(data_root, phase)
        # input_file = os.path.join(data_root, input_file)
        input_file = input_file

        temp_label = json.load(open('image_synthesis/data/imagenet_class_index.json', 'r'))
        self.labels = {}
        for i in range(1000):
            self.labels[temp_label[str(i)][0]] = i
        # print("check labels:", self.labels)
        self.A_paths = []
        self.A_labels = []
        with open(input_file, 'r') as f:
            temp_path = f.readlines()
        for path in temp_path:
            label = self.labels[path.split('/')[0]]
            # print("check path:", path)
            # label = self.labels[path.strip()]
            self.A_paths.append(os.path.join(self.root, path.strip()))
            self.A_labels.append(label)
            # print("check path:", self.A_paths, len(self.A_paths))
            # print("check labels:", self.A_labels, len(self.A_labels))

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        self.phase = phase
        self.inter_negative_samples = inter_negative_samples
        # print("check path:",  len(self.A_paths))
        # print("check labels:",  len(self.A_labels))
        # exit()
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        # if self.transform is not None:
        A = self.transform(A)['image']
        A_label = self.A_labels[index % self.A_size]
        if self.phase == 'train' and self.inter_negative_samples == True:
            # neg_count = 0
            for i in range(10):
                idx = random.randint(0, self.__len__()-1)
                neg_img_path = self.A_paths[idx % self.A_size]
                img = load_img(neg_img_path)
                img = self.transform(image = img)['image']
                if i == 0:
                    neg_img = np.expand_dims(img, axis=0)
                else:
                    img = np.expand_dims(img, axis=0)
                    neg_img = np.concatenate((neg_img, img), axis=0)
            # print("check neg_img:", np.shape(neg_img))
            data = {
                    'image': np.transpose(A.astype(np.float32), (2, 0, 1)),
                    'label': A_label,
                    'negative_img': np.transpose(neg_img.astype(np.float32), (0, 3, 1, 2)),
                }           

        else:
            data = {
                    'image': np.transpose(A.astype(np.float32), (2, 0, 1)),
                    'label': A_label,
                }
        return data
