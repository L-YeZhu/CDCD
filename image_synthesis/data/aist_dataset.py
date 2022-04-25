import torch
import torch.utils.data
import torch.nn.functional as F

from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from pathlib import Path
from librosa.core import load
from librosa.util import normalize
from image_synthesis.utils.misc import instantiate_from_config


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files



class AISTDataset(Dataset):
    def __init__(self, data_root, audio_files, motion_files, video_files, genre_label, augment, segment_length, extra_file_path=None, phase = 'train'):
            
        self.root = os.path.join(data_root, phase)
        self.sampling_rate = 22050
        self.segment_length = self.sampling_rate*segment_length
        self.audio_files = files_to_list(audio_files)
        self.audio_files = [Path(audio_files).parent / x for x in self.audio_files]
        self.augment = True
        self.video_files = files_to_list(video_files)
        self.video_files = [Path(video_files).parent / x for x in self.video_files]
        self.motion_files = files_to_list(motion_files)
        self.genre = np.load(genre_label)
        self.extra_file_path = extra_file_path
        if self.extra_file_path != None:
            with open(extra_file_path, 'r') as f:
                self.extra_music = json.load(f)

 
    def __len__(self):
        return len(self.audio_files)
 
    def __getitem__(self, index):

        # Read audio
        audio_filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(audio_filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # Read video
        video_filename = self.video_files[index]
        video = self.load_img_to_torch(video_filename)

        # Read motion
        motion_filename = self.motion_files[index]
        motion = self.load_motion_to_torch(motion_filename)

        # read genre
        genre = self.genre[index]

        # extra_neagtive_samples if any
        if self.extra_file_path != None:
            extra_music = self.extra_music[index]
            # negative_audio = []
            for i in range(len(extra_music)):
                temp, sampling_rate = self.load_wav_to_torch(extra_music[i])
                if temp.size(0) >= self.segment_length:
                    max_audio_start = temp.size(0) - self.segment_length
                    audio_start = random.randint(0, max_audio_start)
                    temp = temp[audio_start : audio_start + self.segment_length]
                else:
                    temp = F.pad(
                        temp, (0, self.segment_length - temp.size(0)), "constant"
                    ).data

                if i == 0:
                    negative_music = temp.unsqueeze(0)
                else:
                    negative_music = torch.cat((negative_music,temp.unsqueeze(0)),0)
        else:
            negative_music = None

        # print("negative_music:", negative_music.size())



        data = {
                'music': audio.unsqueeze(0),
                'motion': motion,
                'video': video,
                'genre': genre,
                'negative_music': negative_music
        }
        
        return data


    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate


    def load_img_to_torch(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()

    def load_motion_to_torch(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()