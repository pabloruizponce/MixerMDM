import torch

from torch import nn
from models.utils.layers import FFN, VanillaSelfAttention, VanillaCrossAttention

class InfluenceBlockCross(nn.Module):
    """
    New influence block that uses cross attention. 
    In this case the idea is using the self-attention to the individual prediction and
    the cross-attention to the inteaction prediction.
    """
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 **kargs):
        """
        Initialize the block
            :param latent_dim: Latent dimension of the model (input shape)
            :param num_heads: Number of heads in the model
            :param ff_size: Feed forward size of the model (used in the liner layers after the attention)
            :param dropout: Dropout rate of the model
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, m_i, m_I, emb_i=None, emb_I=None, key_padding_mask=None):
        """
        Forward pass of the block
            :param m_i: Input tensor with the motion from the individual model
            :param m_I: Input tensor with the motion from the interaction model
            :param emb_i: Embedding tensor of the condition to the individual model
            :param emb_I: Embedding tensor of the condition to the interaction model
        """
        h1 = self.sa_block(m_i, emb_i, key_padding_mask)
        h1 = h1 + m_i
        h2 = self.ca_block(h1, m_I, emb_I, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb_I)
        out = out + h2
        return out

class Influence(nn.Module):
    """
    Influence modele that predicts the influence of a given model.
    This new version of influence is thinked to be used in the cross attention model.
    Additionally in this case the idea is that the influence is of the shape of an individual motion
    This individual motion is modified by cross-attention with the individual motion from the interaction model
    """
    def __init__(self, input_shape, n_blocks, n_heads, ff_size, mode):
        """
        Initialize the model
            :param input_shape: Shape of the input
            :param n_blocks: Number of blocks in the model
            :param n_heads: Number of heads in the model
            :param ff_size: Feed forward size of the model
            :param mode: Mode of the influence
                * 1: 1 weight (global influence) 
                * 2: 1 weight per timestep (temporal influence)
                * 3: 23 weights (spatial influence) [22 for joints 1 for foot contact]
                * 4: 23 weights per timestep (spatial temporal influence)
        """
        super().__init__()
        self.input_shape = input_shape
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.ff_size = ff_size
        self.mode = mode

        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            self.blocks.append(InfluenceBlockCross(
                latent_dim=self.input_shape,
                num_heads=self.n_heads,
                ff_size=self.ff_size,
            ))

        if self.mode == 1 or self.mode == 2:
            self.out = nn.Linear(self.input_shape, 1)
        elif self.mode == 3 or self.mode == 4:
            self.out = nn.Linear(self.input_shape, 23)
        else:
            raise ValueError("Mode not recognized")
        

    def forward(self, m_i, m_I, cond_i=None, cond_I=None, mask=None):
        """
        Forward pass of the model
            :param m_i: Input tensor with the motion from the individual model
            :param m_I: Input tensor with the motion from the interaction model
            :param cond_i: Input tensor with the condition to the individual model
            :param cond_I: Input tensor with the condition to the interaction model
            :param key_padding_mask: Mask to ignore padding
        """
        # Extracting the batch size and the number of frames
        B, T = m_i.shape[0], m_i.shape[1]

        # Masking
        if mask is None:
            mask = torch.ones(B, T).to(m_i.device)
        else:
            mask = mask[...,0]

        # Inverting the mask to ignore padding
        key_padding_mask = ~(mask > 0.5)

        h_i_prev = m_i
        for block in self.blocks:
            h_i = block(h_i_prev, m_I, cond_i, cond_I, key_padding_mask)
            h_i_prev = h_i

        # Apply mean to have a global average pooling (B, T, D) -> (B, D)
        if self.mode == 1 or self.mode == 3:
            h_i = h_i.mean(dim=1)

        # Final layer
        influence = self.out(h_i)
        influence = nn.functional.sigmoid(influence)
        return influence