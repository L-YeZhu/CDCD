# ------------------------------------------
# CDCD for Dance-to-Music generation
# written By Ye Zhu
# ------------------------------------------

import torch
import math
from torch import nn
from synthesis.utils.misc import instantiate_from_config
import time
import numpy as np
from PIL import Image
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import argparse
from pathlib import Path
import jukebox.utils.dist_adapter as dist

from jukebox.hparams import Hyperparams
from jukebox.utils.torch_utils import empty_cache
from jukebox.utils.audio_utils import save_wav, load_audio
from jukebox.make_models import make_vae_model
from jukebox.utils.sample_utils import split_batch, get_starts
from jukebox.utils.dist_utils import print_once
import fire
import librosa
import soundfile as sf 

from torch.cuda.amp import autocast

def vqvae_param(*args, mode='ancestral', codes_file=None, audio_file=None, prompt_length_in_seconds=None, port=29600, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    ports = [29500, 29600]
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = Hyperparams(**kwargs)
    return device, hps

class D2M(nn.Module):
    def __init__(
        self,
        *,
        content_info={'key': 'music'},
        condition_info_motion={'key': 'motion'},
        condition_info_video={'key': 'video'},
        condition_info_genre={'key': 'genre'},
        negative_music_samples={'key': 'negative_music'},
        hop_level,
        vqvae_load_path,
        content_codec_config,
        condition_codec_config,
        diffusion_config,
        max_vq_len=None,
    ):
        super().__init__()
        self.content_info = content_info
        self.condition_info_motion = condition_info_motion
        self.condition_info_video = condition_info_video
        self.condition_info_genre = condition_info_genre
        self.negative_music_info = negative_music_samples
        self.vqvae = instantiate_from_config(content_codec_config)
        self.condition_codec_motion = instantiate_from_config(condition_codec_config)
        self.transformer = instantiate_from_config(diffusion_config)
        self.truncation_forward = False
        self.vqvae.load_state_dict(torch.load(vqvae_load_path))
        self.vqvae.eval()
        self.max_len = max_vq_len
        if hop_level == 'top':
            self.hop_level = 2
            self.start_level = 2
            self.end_level = 3
        if hop_level == 'middle':
            self.hop_level = 1
            self.start_level = 1
            self.end_level = 2
        if hop_level == 'bottom':
            self.hop_level = 0
            self.start_level = 0
            self.end_level = 1

    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            names = name.split('+')
            params = []
            for n in names:
                try: # the parameters() method is not overwritten for some classes
                    params += getattr(self, name).parameters(recurse=recurse, name=name)
                except:
                    params += getattr(self, name).parameters(recurse=recurse)
            return params

    @property
    def device(self):
        return self.transformer.device

    def get_ema_model(self):
        return self.transformer


    # @torch.no_grad()
    def prepare_condition(self, batch, condition=None):
        cond_key_motion = self.condition_info_motion['key']
        # print("check cond_key:", cond_key_motion)
        cond_motion = batch[cond_key_motion] if condition is None else condition
        # print("check cond:", cond_motion.size())
        if torch.is_tensor(cond_motion):
            cond_motion = cond_motion.to(self.device)
        cond_motion = self.condition_codec_motion(cond_motion)

        cond_key_video = self.condition_info_video['key']
        # print("check cond_key:", cond_key_video)
        cond_video = batch[cond_key_video] if condition is None else condition
        # print("check cond:", cond_video.size())
        if torch.is_tensor(cond_video):
            cond_video = cond_video.to(self.device)      

        cond_key_genre = self.condition_info_genre['key']
        # print("check cond_key:", cond_key_genre)
        cond_genre = batch[cond_key_genre] if condition is None else condition
        # print("check cond:", cond_genre.size())
        if torch.is_tensor(cond_genre):
            cond_genre = cond_genre.to(self.device) 

        # cond = self.condition_codec_motion(cond)
        cond_ = {}
        cond_['condition_motion'] = cond_motion
        cond_['condition_video'] = cond_video
        cond_['condition_genre'] = cond_genre


        # for k, v in cond.items():
        #     v = v.to(self.device) if torch.is_tensor(v) else v
        #     cond_['condition_' + k] = v
        return cond_


    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_content(self, batch, with_mask=False):
        cont_key = self.content_info['key']
        cont = batch[cont_key]
        # print("check cont_key and cont:", cont_key, cont.size())
        if torch.is_tensor(cont):
            cont = cont.to(self.device)

        # print("check cont size:", cont.size())
        gt_xs, zs_code = self.vqvae._encode(cont.transpose(1,2))
        # print("check content token:", zs_code[2].size(), zs_code[2][0,:])
        # zs_cont = []
        # zs_cont.append(zs_code[self.hop_level])
        if self.max_len is not None:
            zs_code[self.hop_level] = zs_code[self.hop_level][:, 0:self.max_len]
        # quantised_xs, out = self.vqvae._decode(zs_cont, start_level=self.start_level, end_level=self.end_level)
        # print("output size:", self.hop_level ,out.size())
        # out = out[0,:,:].squeeze()
        # sf.write('test_sample.wav', out.detach().cpu().numpy(), 22050)

        cont_ = {}
        cont_['content_token'] = zs_code[self.hop_level]
        # print("check content_token size:", cont_['content_token'].size())

        # print("check batch in inference:", batch)
        negative_key = self.negative_music_info['key']
        n_cont = batch[negative_key]
        # print("check n_cont size:", n_cont.size())
        if n_cont != None:
            for i in range(n_cont.size()[0]):
                n_cont_i = n_cont[i,:,:].unsqueeze(1)
                # print("check n_cont_i size:", n_cont_i.size())
                if torch.is_tensor(n_cont_i):
                    n_cont_i = n_cont_i.to(self.device)
                    ngt_xs, nzs_code = self.vqvae._encode(n_cont_i.transpose(1,2))
                    if self.max_len is not None:
                        nzs_code[self.hop_level] = nzs_code[self.hop_level][:, 0:self.max_len]
                    temp_token = nzs_code[self.hop_level]
                    if i == 0:
                        negative_token = temp_token.unsqueeze(0)
                    else:
                        negative_token = torch.cat((negative_token,temp_token.unsqueeze(0)),0)
            cont_['negative_token'] = negative_token
        else:
            cont_['negative_token'] = None
        # print("check negative_token size:", cont_['negative_token'].size()) 

        return cont_




    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_input(self, batch):
        input = self.prepare_condition(batch)
        input.update(self.prepare_content(batch))
        return input

    def p_sample_with_truncation(self, func, sample_type):
        truncation_rate = float(sample_type.replace('q', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            import random
            if random.random() < truncation_rate:
                out = func(out, args[1], args[2], **kwards)
            return out
        return wrapper


    def predict_start_with_truncation(self, func, sample_type):
        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            content_codec = self.content_codec
            save_path = self.this_save_path
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                val, ind = out.topk(k = truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs
            return wrapper
        elif sample_type[-1] == 'r':
            truncation_r = float(sample_type[:-1].replace('top', ''))
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                # notice for different batches, out are same, we do it on out[0]
                temp, indices = torch.sort(out, 1, descending=True) 
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r
                new_temp = torch.full_like(temp3[:,0:1,:], True)
                temp6 = torch.cat((new_temp, temp3), dim=1)
                temp3 = temp6[:,:-1,:]
                temp4 = temp3.gather(1, indices.argsort(1))
                temp5 = temp4.float()*out+(1-temp4.float())*(-70)
                probs = temp5
                return probs
            return wrapper

        else:
            print("wrong sample type")

    @torch.no_grad()
    def generate_content(
        self,
        *,
        batch,
        condition=None,
        filter_ratio = 0,
        temperature = 1.0,
        content_ratio = 0.0,
        replicate=1,
        return_att_weight=False,
        sample_type="top0.85r",
    ):
        self.eval()
        if condition is None:
            condition = self.prepare_condition(batch=batch)
        else:
            condition = self.prepare_condition(batch=None, condition=condition)
        
        if replicate != 1:
            for k in condition.keys():
                if condition[k] is not None:
                    condition[k] = torch.cat([condition[k] for _ in range(replicate)], dim=0)
            
        # content_token = None
        if filter_ratio != 0:
            content = self.prepare_content(batch=batch, with_mask=True)
            content_token = content['content_token']
            # print("check input content_token:", content_token.size())
        else:
            content_token = None


        if len(sample_type.split(',')) > 1:
            if sample_type.split(',')[1][:1]=='q':
                self.transformer.p_sample = self.p_sample_with_truncation(self.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[0][:3] == "top" and self.truncation_forward == False:
            self.transformer.predict_start = self.predict_start_with_truncation(self.transformer.predict_start, sample_type.split(',')[0])
            self.truncation_forward = True

        if len(sample_type.split(',')) == 2 and sample_type.split(',')[1][:4]=='fast':
            trans_out = self.transformer.sample_fast(condition_token=condition['condition_token'],
                                                condition_mask=condition.get('condition_mask', None),
                                                condition_embed=condition.get('condition_embed_token', None),
                                                content_token=content_token,
                                                filter_ratio=filter_ratio,
                                                temperature=temperature,
                                                return_att_weight=return_att_weight,
                                                return_logits=False,
                                                print_log=False,
                                                sample_type=sample_type,
                                                skip_step=int(sample_type.split(',')[1][4:]))

        else:
            trans_out = self.transformer.sample(condition_motion=condition['condition_motion'],
                                            condition_video=condition['condition_video'],
                                            condition_genre= condition['condition_genre'],
                                            condition_mask=condition.get('condition_mask', None),
                                            condition_embed=condition.get('condition_embed_token', None),
                                            content_token=content_token,
                                            filter_ratio=filter_ratio,
                                            temperature=temperature,
                                            return_att_weight=return_att_weight,
                                            return_logits=False,
                                            print_log=False,
                                            sample_type=sample_type)


        # content = self.content_codec.decode(trans_out['content_token'])  #(8,1024)->(8,3,256,256)
        batch_size = trans_out['content_token'].size()[0]
        for k in range(batch_size):
            zs_cont = []
            zs_cont.append(trans_out['content_token'][k,:].unsqueeze(0))
            # print("Chcek input to VQVAE decoder:", trans_out['content_token'][k,:].size(), zs_cont)
            _, out = self.vqvae._decode(zs_cont, start_level=self.start_level, end_level=self.end_level)


        self.train()
        out = {
            'content': out
        }
        # print("check generated output:", out['content'])

        return out

    @torch.no_grad()
    def reconstruct(
        self,
        input
    ):
        if torch.is_tensor(input):
            input = input.to(self.device)
        cont = self.content_codec.get_tokens(input)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        rec = self.content_codec.decode(cont_['content_token'])
        return rec

    @torch.no_grad()
    def sample(
        self,
        batch,
        clip = None,
        temperature = 1.,
        return_rec = True,
        filter_ratio = [0, 0.5, 1.0],
        content_ratio = [1], # the ratio to keep the encoded content tokens
        return_att_weight=False,
        return_logits=False,
        sample_type="normal",
        **kwargs,
    ):
        self.eval()
        condition = self.prepare_condition(batch)
        content = self.prepare_content(batch)
        # print("Check content:", content['content_token'].size())

        content_samples = {'input_music': batch[self.content_info['key']]}
        if return_rec:
            # content_samples['reconstruction_music'] = self.content_codec.decode(content['content_token'])
            batch_size = condition['condition_motion'].size()[0]
            for k in range(batch_size):
                zs_cont = []
                zs_cont.append(content['content_token'])
                _, out = self.vqvae._decode(zs_cont, start_level=self.start_level, end_level=self.end_level)
                # print("check out size:", out.size())
                content_samples['reconstruction_music{}'.format(k)]  = out
                out = out[k,:,:].squeeze()
                recons_sample = 'reconstruction_music' + str(k) + '.wav'
                recons_sample = '/home/zhuye/VQ-Diffusion_d2m/audios/' + recons_sample
                sf.write(recons_sample, out.detach().cpu().numpy(), 22050)

        for fr in filter_ratio:
            for cr in content_ratio:
                num_content_tokens = int((content['content_token'].shape[1] * cr))
                if num_content_tokens < 0:
                    continue
                else:
                    content_token = content['content_token'][:, :num_content_tokens]
                # print("check content_token:", content_token.size())
                if sample_type == 'debug':
                    trans_out = self.transformer.sample_debug(condition_token=condition['condition_token'],
                                                        condition_mask=condition.get('condition_mask', None),
                                                        condition_embed=condition.get('condition_embed_token', None),
                                                        content_token=content_token,
                                                        filter_ratio=fr,
                                                        temperature=temperature,
                                                        return_att_weight=return_att_weight,
                                                        return_logits=return_logits,
                                                        content_logits=content.get('content_logits', None),
                                                        sample_type=sample_type,
                                                        **kwargs)

                else:
                    trans_out = self.transformer.sample(condition_motion=condition['condition_motion'],
                                                        condition_video=condition['condition_video'],
                                                        condition_genre=condition['condition_genre'],
                                                        condition_mask=condition.get('condition_mask', None),
                                                        condition_embed=condition.get('condition_embed_token', None),
                                                        content_token=content_token,
                                                        filter_ratio=fr,
                                                        temperature=temperature,
                                                        return_att_weight=return_att_weight,
                                                        return_logits=return_logits,
                                                        content_logits=content.get('content_logits', None),
                                                        sample_type=sample_type,
                                                       **kwargs)
                batch_size = trans_out['content_token'].size()[0]
                for k in range(batch_size):
                    zs_cont = []
                    zs_cont.append(trans_out['content_token'][k,:].unsqueeze(0))
                    # print("Chcek input to VQVAE decoder:", trans_out['content_token'][k,:].size(), zs_cont)
                    _, out = self.vqvae._decode(zs_cont, start_level=self.start_level, end_level=self.end_level)
                    content_samples['cond1_cont{}_fr{}_music{}'.format(cr, fr, k)] = out
                    sample_audio = 'cond1_cont{}_fr{}_music{}.wav'.format(cr, fr, k)
                    sample_audio = '/home/zhuye/VQ-Diffusion/audios/' + sample_audio
                    sf.write(sample_audio, out.squeeze().detach().cpu().numpy(), 22050)

                if return_att_weight:
                    content_samples['cond1_cont{}_fr{}_music_condition_attention'.format(cr, fr)] = trans_out['condition_attention'] # B x Lt x Ld
                    content_att = trans_out['content_attention']
                    shape = *content_att.shape[:-1], self.content.token_shape[0], self.content.token_shape[1]
                    content_samples['cond1_cont{}_fr{}_music_content_attention'.format(cr, fr)] = content_att.view(*shape) # B x Lt x Lt -> B x Lt x H x W
                if return_logits:
                    content_samples['logits'] = trans_out['logits']
        self.train() 
        output = {'condition_motion': batch[self.condition_info_motion['key']]}
        output = {'condition_video': batch[self.condition_info_video['key']]}
        output = {'condition_genre': batch[self.condition_info_genre['key']]}
        output.update(content_samples)
        return output

    def forward(
        self,
        batch,
        name='none',
        **kwargs
    ):
        input = self.prepare_input(batch)
        output = self.transformer(input, **kwargs)
        return output
