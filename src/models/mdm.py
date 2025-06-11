from models.utils.cfg_sampler import ClassifierFreeSampleModel
from models.utils.gaussian_diffusion import LossType, ModelMeanType, ModelVarType, MotionDiffusion, create_named_schedule_sampler, get_named_beta_schedule, space_timesteps
from models.utils.utils import PositionalEncoding, TimestepEmbedder
import numpy as np
import torch
import torch.nn as nn
import clip

class MDM(nn.Module):
    def __init__(self, cfg, num_frames=300, sampling_strategy="ddim50"):
        super().__init__()

        self.cfg = cfg

        # Model parameters
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP
        self.n_joints = None

        # Model
        self.model = MDMDenoiser(
            self.n_joints, 
            self.nfeats, 
            self.latent_dim, 
            self.ff_size, 
            self.num_layers, 
            self.num_heads, 
            self.dropout,
            self.activation
        )

        # Text encoders
        self.embed_text = nn.Linear(512, self.latent_dim)
        self.clip_version = None
        self.clip_model = self.load_and_freeze_clip(self.clip_version)

        # Diffusion Model
        # Parameters
        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampling_strategy = sampling_strategy
        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)
        timestep_respacing = [self.diffusion_steps]
        # Model
        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            mode="individual"
        )
        
        # Sampler
        self.sampler = cfg.SAMPLER
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, cond_mask_prob=0.1):
        """
        Function to mask the conditions with a given probability.
        Used for Classifier Free Guidance in the diffusion model
            :param cond: Conditions to mask
            :param cond_mask_prob: Probability of masking the conditions
            :return: Masked conditions and the mask
        """
        B = cond.shape[0]
        # Mask the conditions with a given probability
        if cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(B, device=cond.device) * cond_mask_prob).view([B]+[1]*len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), (1. - mask)
        else:
            return cond, None

    def text_process(self, batch, mode, text_name="text", out_name="cond"):
        # raw_text - list (batch_size length) of strings with input text prompts
        raw_text = batch[text_name]
        device = next(self.parameters()).device
        max_text_len = 20  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        
        cond = self.clip_model.encode_text(texts).float()
        cond = self.embed_text(cond)
        batch[out_name] = cond
        
        return 
    
    
    def generate_src_mask(self, T, length):
        """
        Funtion to generate a mask for an interaction motion.
        It sets to 0 a mask for the interaction motion after the length of the motion is completed
            :param T: Total length of the motion in the batch
            :param length: Length of the motion in the batch
            :return: Binary Mask for the interaction motion
        """
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask
    

    def generate_cond(self, batch):
        """
        Function to generate encode all the text conditions and store them in the batch
            :param batch: Batch with the text conditions
            :return: Encoded conditions
        """
        self.text_process(batch, None, "text", "cond_individual_individual1")
        cond = batch["cond_individual_individual1"]
        return cond 
    

    def compute_loss(self, batch):
        """
        Computing the losses of the mixing model using the traditional pipeline for training diffusion models
            :param batch: Batch with the data
            :mode: Define if we are training the generator or the discriminator
            :i_loss_factor: Factor to weight the influence loss
            :I_loss_factor: Factor to weight the influence loss
            :return: Total loss and all the losses
        """
        # Generate the encoded representation of the conditions from the raw text
        cond = self.generate_cond(batch)
        B = cond.shape[0]

        # Get the ground thruth motions
        x_start = batch["motions"]

        # Mask the conditions
        if cond is not None:
            cond, cond_mask = self.mask_cond(cond, 0.1)

        # Generate the masks for the motion input
        seq_mask = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"]).to(x_start.device)
        
        # Sample random timesteps to compute the losses in specific timesteps
        t, _ = self.sampler.sample(B, x_start.device)

        model = self.model

        output = self.diffusion.training_losses(
            model=model,
            x_start=x_start,
            t=t,
            mask=seq_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=cond_mask,
            model_kwargs={
                "mask":seq_mask,
                "cond":cond,
            },
        )
        return output["total"], output


    def forward(self, batch):
        
        # Generate the encoded representation of the conditions from the raw text
        cond = self.generate_cond(batch)

        B = cond.shape[0]
        T = batch["motion_lens"][0]

        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            mode = "individual"
        )

        self.cfg_model = ClassifierFreeSampleModel(self.model, self.cfg_weight)
        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B, T, self.nfeats),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "mask":None,
                "cond":cond,
            },
            x_start=None
        )

        return {"output":output}

    def forward_test(self, batch):
        return self.forward(batch)
    

class MDMDenoiser(nn.Module):
    def __init__(self, njoints, nfeats, latent_dim, ff_size, num_layers, num_heads, dropout, activation):
        super().__init__()

        self.text_dim = 256
        self.njoints = njoints
        self.nfeats = nfeats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.input_feats = self.nfeats

        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=self.num_layers
        )
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.output_process = OutputProcess(
            self.input_feats, 
            self.latent_dim, 
            self.njoints,
            self.nfeats
        )

    def forward(self, x, timesteps, cond=None, mask=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, nframes, nfeats  = x.shape
        cond += self.embed_timestep(timesteps)  # [bs, d]
        cond = cond.unsqueeze(1)

        x = self.input_process(x)

        if mask is not None:
            mask = mask[...,0]
        else:
            mask = torch.ones(bs, nframes).to(x.device)
        
        # Add an additional frame at the beginning of the sequence
        mask = torch.cat([torch.ones(bs, 1).to(x.device), mask], dim=1)
        key_padding_mask = ~(mask > 0.5)

        # adding the timestep embed
        xseq = torch.cat((cond, x), axis=1)  # [bs, seqlen+1, d]
        xseq = self.sequence_pos_encoder(xseq)  # [bs, seqlen+1, d]
        output = self.seqTransEncoder(xseq, src_key_padding_mask=key_padding_mask)[:,1:,:]  # , src_key_padding_mask=~maskseq)  # [bs, seqlen, d]
        output = self.output_process(output)  # [bs, nframes, nfeats]
        return output        

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nframes, nfeats = x.shape
        x = self.poseEmbedding(x)  # [bs, nframes, latent_dim]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        bs, nframes, d = output.shape
        output = self.poseFinal(output)  # [bs, seqlen, bs, 262]
        return output
