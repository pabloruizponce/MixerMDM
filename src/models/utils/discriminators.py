import torch
import torch.nn as nn

from models.utils.blocks import TransformerBlockSimple
from models.utils.utils import PositionalEncoding, TimestepEmbedder

class DiscriminatorTransfomer(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim,
                 num_frames,
                 ff_size,
                 num_layers,
                 num_heads,
                 dropout=0.1,
                 activation="gelu",
                 **kargs):
        super().__init__()

        # Model Parameters
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.text_emb_dim = 768

        # Time Embedding
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        # Transformer Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlockSimple(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))

        # Output Module
        self.out = nn.Linear(self.latent_dim, 1)


    def forward(self, x, timesteps, mask=None, cond=None):
        """
        x: B, T, D
        """
        # Extracting the batch size and the number of frames
        B, T = x.shape[0], x.shape[1]

        # Embedding the condition and the input
        cond_emb = self.embed_timestep(timesteps) + self.text_embed(cond)
        x_emb = self.motion_embed(x)
        h_prev = self.sequence_pos_encoder(x_emb)

        # Masking
        if mask is None:
            mask = torch.ones(B, T).to(x.device)
        else:
            mask = mask[...,0]

        # Inverting the mask to ignore padding
        key_padding_mask = ~(mask > 0.5)

        # Forward pass through the transformer blocks
        for i,block in enumerate(self.blocks):
            h = block(h_prev, cond_emb, key_padding_mask)
            h_prev = h

        # Output Module
        output = self.out(h)
        return output