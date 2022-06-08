import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np
from .base_embedding import BaseEmbedding


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConv2d(*arg, **kwargs):
    return weight_norm(nn.Conv2d(*arg, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def WNConvTranspose2d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose2d(*args, **kwargs))


class MotionEmbedding(BaseEmbedding):
    def __init__(self, 
                 num_embed=1000,
                 embed_dim=512,
                 identity=False,
                 trainable=True,
        ):
        super().__init__()
        self.identity = identity
        self.trainable = trainable
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        if self.identity == False:
        
            self.emb = nn.Embedding(self.num_embed, embed_dim)
            self._set_trainable()

    def forward(self, index, **kwargs):
        """
        index: B x L, index
        mask: B x L, bool type. The value of False indicating padded index
        """
        if self.identity == True:
            return index
        else:
            emb = self.emb(index).unsqueeze(1)
            return emb

