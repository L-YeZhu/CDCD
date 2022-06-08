import torch
import torch.nn as nn
from .base_embedding import BaseEmbedding

class GenreEmbedding(BaseEmbedding):
    def __init__(self, 
                 num_embed=10,
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
        
            self.genre_embed = nn.Embedding(self.num_embed, embed_dim)
            self._set_trainable()

    def forward(self, genre, **kwargs):
        """
        genre: B x L, index (one-hot vector)
        mask: B x L, bool type. The value of False indicating padded index
        """
        genre_idx = genre.nonzero(as_tuple=True)[1]
        genre_emb = self.genre_embed(genre_idx).unsqueeze(1)
        if self.identity == True:
            return genre_idx
        else:
            # emb = self.emb(index).unsqueeze(1)
            return genre_emb

