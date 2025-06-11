from models.utils.discriminators import DiscriminatorTransfomer
from models.utils.influence import Influence
from models.intergen import InterGen
from models.mdm import MDM
from models.utils.cfg_sampler import ClassifierFreeSampleModelX2
from models.utils.gaussian_diffusion import LossType, ModelMeanType, ModelVarType, MixerDiffusion, create_named_schedule_sampler, get_named_beta_schedule, space_timesteps
import torch
import clip

from torch import nn
from models.in2in import in2IN
from models.utils.utils import PositionalEncoding, TimestepEmbedder, set_requires_grad
from utils.configs import get_config
from utils.utils import MotionNormalizerTorch, MotionNormalizerTorchHML3D
from utils.alignment import smpl_to_ih, ih_to_smpl, align_motions


class MixerMDM(nn.Module):
    def __init__(self, cfg, num_frames=300, sampling_strategy="ddim50", store_influence=True, align=True):
        super().__init__()
        self.cfg = cfg
        self.cfg_model1 = get_config(cfg.MODEL1)
        self.cfg_model2 = get_config(cfg.MODEL2)

        # Alingment flag
        self.align = align

        # Variable to determine if we load the models
        self.store_influence = store_influence

        # Add the 2 models that we want to combine
        if self.cfg_model1.NAME == "MDM":
            self.model1 = MDM(self.cfg_model1)
        elif self.cfg_model1.NAME == "in2INind":
            self.model1 = in2IN(self.cfg_model1, mode="individual") # Model 1

        if self.cfg_model2.NAME == "InterGen":
            self.model2 = InterGen(self.cfg_model2)
        elif self.cfg_model2.NAME == "in2IN":
            self.model2 = in2IN(self.cfg_model2, mode="interaction") # Model 2

        # Load the models
        if self.cfg_model1.NAME == "MDM":
            ckpt = torch.load(self.cfg_model1.CHECKPOINT, map_location="cpu")
            for k in list(ckpt["state_dict"].keys()):
                ckpt["state_dict"][k[6:]] = ckpt["state_dict"].pop(k)
            self.model1.load_state_dict(ckpt["state_dict"], strict=True)
        elif self.cfg_model1.NAME == "in2INind":
            self.model1.load_state_dict(torch.load(self.cfg_model1.CHECKPOINT), strict=True)


        if self.cfg_model2.NAME == "InterGen":
            ckpt = torch.load(self.cfg_model2.CHECKPOINT, map_location="cpu")
            for k in list(ckpt["state_dict"].keys()):
                if "model" in k:
                    ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
            self.model2.load_state_dict(ckpt["state_dict"], strict=True)
        elif self.cfg_model2.NAME == "in2IN":
            self.model2.load_state_dict(torch.load(self.cfg_model2.CHECKPOINT), strict=True)

        # Freeze the models
        set_requires_grad(self.model1, False)
        set_requires_grad(self.model2, False)
        self.model1.eval()
        self.model2.eval()

        # Load the denoisers
        self.denoiser1 = self.get_denoiser(self.model1)
        self.denoiser2 = self.get_denoiser(self.model2)

        # Load the condition preprocessing functions
        self.cond_pre_func1 = self.get_cond_preprocessing(self.model1)
        self.cond_pre_func2 = self.get_cond_preprocessing(self.model2)

        # Model parameters

        #  Check if they key GENERATOR exits
        if "GENERATOR" in cfg and "DISCRIMINATOR" in cfg:
            self.g_nfeats = cfg.GENERATOR.INPUT_DIM
            self.g_latent_dim = cfg.GENERATOR.LATENT_DIM
            self.g_ff_size = cfg.GENERATOR.FF_SIZE
            self.g_num_layers = cfg.GENERATOR.NUM_LAYERS
            self.g_num_heads = cfg.GENERATOR.NUM_HEADS
            self.g_dropout = cfg.GENERATOR.DROPOUT

            self.d_nfeats = cfg.DISCRIMINATOR.INPUT_DIM
            self.d_latent_dim = cfg.DISCRIMINATOR.LATENT_DIM
            self.d_ff_size = cfg.DISCRIMINATOR.FF_SIZE
            self.d_num_layers = cfg.DISCRIMINATOR.NUM_LAYERS
            self.d_num_heads = cfg.DISCRIMINATOR.NUM_HEADS
            self.d_dropout = cfg.DISCRIMINATOR.DROPOUT
            
            self.nfeats = self.g_nfeats
        else:
            self.nfeats = cfg.INPUT_DIM
            self.latent_dim = cfg.LATENT_DIM
            self.ff_size = cfg.FF_SIZE
            self.num_layers = cfg.NUM_LAYERS
            self.num_heads = cfg.NUM_HEADS
            self.dropout = cfg.DROPOUT


        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP
        self.cfg_mixing_weight = cfg.CFG_WEIGHT
        self.text_dim = 768
        self.mixing_mode = cfg.MIXING_MODE

        # Diffusion Model
        # Parameters
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampling_strategy = sampling_strategy
        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)
        timestep_respacing = [self.diffusion_steps]
        # Model
        self.diffusion = MixerDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )
        
        # Sampler
        self.sampler = cfg.SAMPLER
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

        # Mixing model (denoiser)
        self.force_influnce_val = cfg.FORCE_INFLUENCE_VAL

        if "GENERATOR" in cfg:
            self.mixing = Mixer(
                denoiser1=self.denoiser1, 
                denoiser2=self.denoiser2, 
                nfeats=self.g_nfeats, 
                latent_dim=self.g_latent_dim, 
                ff_size=self.g_ff_size,
                text_dim=self.text_dim,
                n_blocks=self.g_num_layers,
                n_heads=self.g_num_heads,
                mixing_mode=self.mixing_mode,
                store_influence=self.store_influence,
                force_influence_val=self.force_influnce_val,
                align=self.align
            )
        else:
            self.mixing = Mixer(
                denoiser1=self.denoiser1, 
                denoiser2=self.denoiser2, 
                nfeats=self.nfeats, 
                latent_dim=self.latent_dim, 
                ff_size=self.ff_size,
                text_dim=self.text_dim,
                n_blocks=self.num_layers,
                n_heads=self.num_heads,
                mixing_mode=self.mixing_mode,
                store_influence=self.store_influence,
                force_influence_val=self.force_influnce_val,
                align=self.align
            )

        # Discriminators
        if "DISCRIMINATOR" in cfg:
            self.discriminator_i = DiscriminatorTransfomer(
                input_feats=self.d_nfeats,
                latent_dim=self.d_latent_dim,
                num_frames=num_frames,
                ff_size=self.d_ff_size,
                num_layers=self.d_num_layers,
                num_heads=self.d_num_heads,
                dropout=self.d_dropout,
                activation=self.activation,
            )

            self.discriminator_I = DiscriminatorTransfomer(
                input_feats=self.d_nfeats*2,
                latent_dim=self.d_latent_dim,
                num_frames=num_frames,
                ff_size=self.d_ff_size,
                num_layers=self.d_num_layers,
                num_heads=self.d_num_heads,
                dropout=self.d_dropout,
                activation=self.activation,
            )
        else:
            self.discriminator_i = DiscriminatorTransfomer(
                input_feats=self.nfeats,
                latent_dim=self.latent_dim,
                num_frames=num_frames,
                ff_size=self.ff_size,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                activation=self.activation,
            )

            self.discriminator_I = DiscriminatorTransfomer(
                input_feats=self.nfeats*2,
                latent_dim=self.latent_dim,
                num_frames=num_frames,
                ff_size=self.ff_size,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                activation=self.activation,
            )

        # CLIP model
        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        # Freeze the clip model (backbone)
        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        # Change references of CLIP models on other models to reduce space

        # Temporally comented to avoid problems with the swap evaluation
        #if self.cfg_model1.NAME == "in2INind":
        #    print("Changing references of CLIP models on other models to reduce space")
        #    self.model1.token_embedding = self.token_embedding
        #    self.model1.clip_transformer = self.clip_transformer
        #    self.model1.ln_final = self.ln_final
        #    self.model1.dtype = self.dtype
        #    torch.cuda.empty_cache()

        self.model2.token_embedding = self.token_embedding
        self.model2.clip_transformer = self.clip_transformer
        self.model2.ln_final = self.ln_final
        self.model2.dtype = self.dtype
        torch.cuda.empty_cache()

        # Additional encoder to CLIP representation for more menaingful conditions
        # This is done because the condtion space in CLIP and in our problem is not the same
        # Because of that we can use a transformer to encode the CLIP representation to a more meaningful one 
        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.text_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )   
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer,
            num_layers=2
        )
        self.clip_ln = nn.LayerNorm(self.text_dim)


    def get_denoiser(self, model):
        """
        Function to get the specific denoiser from each of the used models
            :param model: Model from which we want to get the denoiser
            :return: Denoiser of the model
        """
        if model.cfg.NAME == "MDM":
            return model.model
        elif model.cfg.NAME == "in2INind":
            return model.decoder.net_individual
        elif model.cfg.NAME == "in2IN":
            return model.decoder.net_interaction
        elif model.cfg.NAME == "InterGen":
            return model.decoder.net

    
    def get_cond_preprocessing(self, model):
        """
        Function to get the specific condition preprocessing function from each of the used models
            :param model: Model from which we want to get the condition preprocessing function
            :return: Condition preprocessing function of the model
        """
        return model.text_process

    def text_process(self, batch, text_name="text", out_name="cond"):
        """
        Function to encode text conditions into a latent representation
            :param batch: Batch with the text to process
            :param text_name: Name of the text in the batch
            :param out_name: Name of the output in the batch
        """
        
        # Extract the text an the device
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch[text_name]

        # Process the text to get the CLIP representation
        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self.dtype)

        # Encode the CLIP representation to a more meaningful one
        out = self.clipTransEncoder(clip_out)
        out = self.clip_ln(out)

        # Return the encoded representation
        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        batch[out_name] = cond

        return batch
    
    def generate_cond(self, batch):
        """
        Function to generate encode all the text conditions and store them in the batch
            :param batch: Batch with the text conditions
            :return: Encoded conditions
        """

        # Preprocess the text conditions with the specific functions for each models
        self.cond_pre_func1(batch, "individual", "text_individual1","cond_individual_individual1")
        self.cond_pre_func1(batch, "individual", "text_individual2","cond_individual_individual2")
        self.cond_pre_func2(batch, "interaction", "text_individual1","cond_interaction_individual1")
        self.cond_pre_func2(batch, "interaction", "text_individual2","cond_interaction_individual2")
        # Distinc cases for different models
        if "text_interaction" in batch:
            self.cond_pre_func2(batch, "interaction", "text_interaction","cond_interaction")
        elif "text" in batch:
            self.cond_pre_func2(batch, "interaction", "text","cond_interaction")

        # Preprocess text condtions with the specif encoder os the mixing model
        self.text_process(batch, "text_individual1", "cond_influence_individual1")
        self.text_process(batch, "text_individual2", "cond_influence_individual2")
        if "text_interaction" in batch:
            self.text_process(batch, "text_interaction", "cond_influence_interaction")
        elif "text" in batch:
            self.text_process(batch, "text", "cond_influence_interaction")

        # Concatenate all the conditions into a single tensor
        cond = torch.cat(
            [
                batch["cond_interaction"], 
                batch["cond_interaction_individual1"], 
                batch["cond_interaction_individual2"], 
                batch["cond_individual_individual1"], 
                batch["cond_individual_individual2"],
                batch["cond_influence_interaction"],
                batch["cond_influence_individual1"],
                batch["cond_influence_individual2"]
            ], 
            dim=1
        )

        return cond 

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
        
    def compute_loss(self, batch, mode, i_loss_factor, I_loss_factor, l1):
        """
        Computing the losses of the mixing model using the traditional pipeline for training diffusion models
            :param batch: Batch with the data
            :mode: Define if we are training the generator or the discriminator
            :i_loss_factor: Factor to weight the influence loss
            :I_loss_factor: Factor to weight the influence loss
            :return: Total loss and all the losses
        """
        # Set mixing model to train mode
        self.mixing.mode = "train"

        # Depending on the training mode, we freeze the generator or the discriminator
        if mode == "generator":
            self.mixing.train()
            self.clipTransEncoder.train(),
            self.clip_ln.train()
            self.discriminator_I.eval()
            self.discriminator_i.eval()

            self.mixing.requires_grad_(True)
            # I have to take a lot of care with this shit
            self.mixing.denoiser1.requires_grad_(False)
            self.mixing.denoiser2.requires_grad_(False)
            self.positional_embedding.requires_grad_(True)
            self.clipTransEncoder.requires_grad_(True)
            self.clip_ln.requires_grad_(True)
            self.discriminator_i.requires_grad_(False)
            self.discriminator_I.requires_grad_(False)
        elif mode == "discriminator":
            self.mixing.eval()
            self.clipTransEncoder.eval()
            self.clip_ln.eval()
            self.discriminator_i.train()
            self.discriminator_I.train()

            self.mixing.requires_grad_(False)
            self.positional_embedding.requires_grad_(False)
            self.clipTransEncoder.requires_grad_(False)
            self.clip_ln.requires_grad_(False)
            self.discriminator_i.requires_grad_(True)
            self.discriminator_I.requires_grad_(True)
        else:
            raise ValueError("Mode not recognized")

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

        # Store influence
        if self.store_influence:
            self.mixing.history_influence_i1 = []
            self.mixing.history_influence_i2 = []

        # Compute the loss of the diffusion model with the mixing model
        output = self.diffusion.training_losses(
            generator=self.mixing,
            discriminator_i=self.discriminator_i,
            discriminator_I=self.discriminator_I,
            mode=mode,
            x_start=x_start,
            t=t,
            mask=seq_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=cond_mask,
            i_loss_factor=i_loss_factor,
            I_loss_factor=I_loss_factor,
            l1=l1,
            model_kwargs={
                "mask":seq_mask,
                "cond":cond,
            },
        )
        if self.store_influence:
            influence_history = {
                "influence_i1": self.mixing.history_influence_i1, 
                "influence_i2": self.mixing.history_influence_i2
            }
        else:
            influence_history = None

        # Return the total loss and all the lossed dictionary
        if mode == "generator":
            return output["generator_total"], output, influence_history
        elif mode == "discriminator":
            return output["discriminator_total"], output, influence_history

    def forward(self, batch):
        """
        Forward pass of the model. It computes the full denoising chain using the mixing model
            :param batch: Batch with the data
            :return: Output of the model
        """
        # Fix the model into evaluation mode
        self.mixing.mode = "eval"

        # Generate the encoded representation of the conditions from the raw text
        cond = self.generate_cond(batch)
        B = cond.shape[0]
        T = batch["motion_lens"][0]

        # Reseting the history of the influence on each inference
        self.mixing.history_influence_i1 = []
        self.mixing.history_influence_i2 = []
        self.mixing.history_out1 = []
        self.mixing.history_out2 = []
        self.mixing.history_out_influenced = []

        # Define the timenstep respacing to the samplin strategy for DDIM sampling
        timestep_respacing= self.sampling_strategy

        # Defining a new diffusion model with the specific sampling strategy
        self.diffusion_test = MixerDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )

        # Use Classifier Free Guidance in the diffusion model for the inference
        self.cfg_model = ClassifierFreeSampleModelX2(self.mixing, self.cfg_mixing_weight)

        # Compute the output of the diffusion model with the mixing model
        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B, T, self.nfeats*2),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "mask":None,
                "cond":cond,
            },
            x_start=None
        )

        # Return a dictionary with the output and the influences
        return {
            "output":output, 
            "influence_i1": self.mixing.history_influence_i1, 
            "influence_i2": self.mixing.history_influence_i2,
            "out1": self.mixing.history_out1,
            "out2": self.mixing.history_out2,
            "out_influenced": self.mixing.history_out_influenced
        }
    
    def forward_test(self, batch):
        """
        Forward pass of the model. It computes the full denoising chain using the mixing model
            :param batch: Batch with the data
            :return: Output of the model
        """
        # Fix the model into evaluation mode
        self.mixing.mode = "eval_intermediate"

        # Generate the encoded representation of the conditions from the raw text
        cond = self.generate_cond(batch)
        B = cond.shape[0]
        T = batch["motion_lens"][0]

        # Reseting the history of the influence on each inference
        self.mixing.history_influence_i1 = []
        self.mixing.history_influence_i2 = []

        # Define the timenstep respacing to the samplin strategy for DDIM sampling
        timestep_respacing= self.sampling_strategy

        # Defining a new diffusion model with the specific sampling strategy
        self.diffusion_test = MixerDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )

        # Use Classifier Free Guidance in the diffusion model for the inference
        self.cfg_model = ClassifierFreeSampleModelX2(self.mixing, self.cfg_mixing_weight)

        # Compute the output of the diffusion model with the mixing model
        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B, T, self.nfeats*2),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "mask":None,
                "cond":cond,
            },
            x_start=None
        )

        # Return a dictionary with the output and the influences
        return {
            "output":output, 
            "influence_i1": self.mixing.history_influence_i1, 
            "influence_i2": self.mixing.history_influence_i2,
        }

class Mixer(nn.Module):

    def __init__(self, denoiser1, denoiser2, nfeats, latent_dim, ff_size, text_dim, n_blocks, n_heads, mixing_mode, store_influence=False, force_influence_val=None, mode="train", align=True):
        super().__init__()
        
        # Model parameters
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.text_dim = text_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.mixing_mode = mixing_mode
        self.mode = mode
        self.align = align 

        # Loading the denoisers
        self.denoiser1 = denoiser1
        self.denoiser2 = denoiser2
        

        # Influence model version 2
        self.force_influence_val = force_influence_val
        self.influence = Influence(
            input_shape=self.latent_dim,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            ff_size=self.ff_size,
            mode=self.mixing_mode,
        )

        # Influence history for later visualizations
        self.store_influence = store_influence
        if self.store_influence:
            self.history_influence_i1 = []
            self.history_influence_i2 = []

        # Store model intermediate outputs
        if self.mode == "eval":
            self.history_out1 = []
            self.history_out2 = []
            self.history_out_influenced = []

        # Positional Encoding and Timestep Embedding
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.nfeats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_dim, self.latent_dim)

        # Normalizers 
        self.normalizer_model1 = MotionNormalizerTorchHML3D()
        self.normalizer_model2 = MotionNormalizerTorch()


    def forward(self, x1, timesteps, cond=None, mask=None, x2=None):

        B, T = x1.shape[:2]

        # Getting the two motions from the interaction model
        x1 = x1.float()
        x1_i1 = x1[:, :, :self.nfeats].float()
        x1_i2 = x1[:, :, self.nfeats:].float()
        x2 = x2.float()
            
        # Getting the conditions
        # Firstly we get the conditions encoded from the pre-trained models
        cond1_1 = cond[:, self.denoiser2.text_dim*3:self.denoiser2.text_dim*3+self.denoiser1.text_dim*1]
        cond1_2 = cond[:, self.denoiser2.text_dim*3+self.denoiser1.text_dim*1:self.denoiser2.text_dim*3+self.denoiser1.text_dim*2]
        cond2 = cond[:, :self.text_dim*3]

        # Then getting the raw condtions and embedding them for the mixer
        cond_I = self.text_embed(cond[:, self.denoiser2.text_dim*3+self.denoiser1.text_dim*2:self.denoiser2.text_dim*4+self.denoiser1.text_dim*2])
        cond_I = self.embed_timestep(timesteps) + cond_I
        cond_i1 = self.text_embed(cond[:, self.denoiser2.text_dim*4+self.denoiser1.text_dim*2:self.denoiser2.text_dim*5+self.denoiser1.text_dim*2])
        cond_i1 = self.embed_timestep(timesteps) + cond_i1
        cond_i2 = self.text_embed(cond[:, self.denoiser2.text_dim*5+self.denoiser1.text_dim*2:self.denoiser2.text_dim*6+self.denoiser1.text_dim*2])
        cond_i2 = self.embed_timestep(timesteps) + cond_i2

        # Calculate the predictions of the frozen models (individual)
        out1_1 = self.denoiser1(x1_i1, timesteps, cond=cond1_1, mask=mask)
        out1_2 = self.denoiser1(x1_i2, timesteps, cond=cond1_2, mask=mask)
        out2 = self.denoiser2(x2, timesteps, cond=cond2, mask=mask)


        # Denormalizing to use the original space of the motions in the influence models
        out1_1 = self.normalizer_model1.backward(out1_1)
        out1_2 = self.normalizer_model1.backward(out1_2)
        out1 = torch.cat([out1_1, out1_2], dim=-1)
        out2 = self.normalizer_model2.backward(out2.reshape(B, T, 2, -1)).reshape(B, T, -1)

        out1_1 = out1[...,:self.nfeats]
        out1_2 = out1[...,self.nfeats:]
        out2_1 = out2[...,:self.nfeats]
        out2_2 = out2[...,self.nfeats:]

        # Align the motions of the individual denoirser to the trajectories of the interaction denoiser
        if self.align:
            out1_1 = ih_to_smpl(out1_1)
            out1_2 = ih_to_smpl(out1_2)
            out2_1 = ih_to_smpl(out2_1)
            out2_2 = ih_to_smpl(out2_2)
            _, out1_1 = align_motions(out2_1, out1_1, mask)
            _, out1_2 = align_motions(out2_2, out1_2, mask)
            out1_1 = smpl_to_ih(out1_1)
            out1_2 = smpl_to_ih(out1_2)
            out2_1 = smpl_to_ih(out2_1)
            out2_2 = smpl_to_ih(out2_2)
        
        out1 = torch.cat([out1_1, out1_2], dim=-1)
        out2 = torch.cat([out2_1, out2_2], dim=-1)
        out1_1 = out1[...,:self.nfeats]
        out1_2 = out1[...,self.nfeats:]
        out2_1 = out2[...,:self.nfeats]
        out2_2 = out2[...,self.nfeats:]

        # Embed output motions
        out1_1_emb = self.motion_embed(out1_1)
        out1_2_emb = self.motion_embed(out1_2)

        out2_1_emb = self.motion_embed(out2_1)
        out2_2_emb = self.motion_embed(out2_2)

        # Include positional encoding
        out1_1_emb = self.sequence_pos_encoder(out1_1_emb)
        out1_2_emb = self.sequence_pos_encoder(out1_2_emb)
        out2_1_emb = self.sequence_pos_encoder(out2_1_emb)
        out2_2_emb = self.sequence_pos_encoder(out2_2_emb)

        # Calculate the influence for each one of the individuals
        influence_i1 = self.influence(out1_1_emb, out2_1_emb, cond_i1, cond_I, mask)
        influence_i2 = self.influence(out1_2_emb, out2_2_emb, cond_i2, cond_I, mask)

        # Duplicate the influence to match the shape of the output
        if self.mixing_mode == 1:
            influence_i1 = influence_i1.unsqueeze(1).expand(-1, out1_1.shape[1], -1)
            influence_i2 = influence_i2.unsqueeze(1).expand(-1, out1_2.shape[1], -1)
        elif self.mixing_mode == 2:
            influence_i1 = influence_i1
            influence_i2 = influence_i2
        elif self.mixing_mode == 3:
            
            influence_i1 = influence_i1.unsqueeze(1).expand(-1, out1_1.shape[1], -1)
            influence_i2 = influence_i2.unsqueeze(1).expand(-1, out1_2.shape[1], -1)

            influence_i1_j = influence_i1[..., :22]
            influence_i1_j = influence_i1_j.repeat_interleave(3, dim=-1)
            influence_i1_v = influence_i1_j
            influence_i1_r = influence_i1[..., :21]
            influence_i1_r = influence_i1_r.repeat_interleave(6, dim=-1)
            influence_i1_f = influence_i1[..., 22:]
            influence_i1_f =  influence_i1_f.expand(-1, -1, 4)
            influence_i1 = torch.cat([influence_i1_j, influence_i1_v, influence_i1_r, influence_i1_f], dim=-1)

            influence_i2_j = influence_i2[..., :22]
            influence_i2_j = influence_i2_j.repeat_interleave(3, dim=-1)
            influence_i2_v = influence_i2_j
            influence_i2_r = influence_i2[..., :21]
            influence_i2_r = influence_i2_r.repeat_interleave(6, dim=-1)
            influence_i2_f = influence_i2[..., 22:]
            influence_i2_f =  influence_i2_f.expand(-1, -1, 4)
            influence_i2 = torch.cat([influence_i2_j, influence_i2_v, influence_i2_r, influence_i2_f], dim=-1)
        elif self.mixing_mode == 4:
            influence_i1_j = influence_i1[..., :22]
            influence_i1_j = influence_i1_j.repeat_interleave(3, dim=-1)
            influence_i1_v = influence_i1_j
            influence_i1_r = influence_i1[..., :21]
            influence_i1_r = influence_i1_r.repeat_interleave(6, dim=-1)
            influence_i1_f = influence_i1[..., 22:]
            influence_i1_f =  influence_i1_f.expand(-1, -1, 4)
            influence_i1 = torch.cat([influence_i1_j, influence_i1_v, influence_i1_r, influence_i1_f], dim=-1)

            influence_i2_j = influence_i2[..., :22]
            influence_i2_j = influence_i2_j.repeat_interleave(3, dim=-1)
            influence_i2_v = influence_i2_j
            influence_i2_r = influence_i2[..., :21]
            influence_i2_r = influence_i2_r.repeat_interleave(6, dim=-1)
            influence_i2_f = influence_i2[..., 22:]
            influence_i2_f =  influence_i2_f.expand(-1, -1, 4)
            influence_i2 = torch.cat([influence_i2_j, influence_i2_v, influence_i2_r, influence_i2_f], dim=-1)
        else:
            raise ValueError("Mixing mode not recognized")    

        # Force the influence to a specific value
        if self.force_influence_val is not None:
            influence_i1 = torch.ones_like(influence_i1) * self.force_influence_val
            influence_i2 = torch.ones_like(influence_i2) * self.force_influence_val

        # Store the predictions of the influence for later visualizations
        if self.store_influence:
            self.history_influence_i1.append(influence_i1)
            self.history_influence_i2.append(influence_i2)   

        # Mix the outputs of the models with the influences
        out_i1_influenced = out2_1 + influence_i1 * (out1_1 - out2_1)
        out_i2_influenced = out2_2 + influence_i2 * (out1_2 - out2_2)
        out_influenced = torch.cat([out_i1_influenced, out_i2_influenced], dim=-1)

        if self.mode == "train":
            return out_influenced, out1, out2
        elif self.mode == "eval":
            self.history_out1.append(out1)
            self.history_out2.append(out2)
            self.history_out_influenced.append(out_influenced)
            return out_influenced
        elif self.mode == "eval_intermediate":
            return out_influenced 