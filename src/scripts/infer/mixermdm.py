import sys
sys.path.append(sys.path[0] + r"/../../")

import copy
import torch
import os.path
import argparse
import numpy as np

from collections import OrderedDict
from utils.configs import get_config
from lightning import LightningModule
from utils.plot import plot_3d_motion, plot_influence
from utils.utils import MotionNormalizer
from scipy.ndimage import gaussian_filter1d
from models.mixermdm import MixerMDM
from utils.paramUtil import HML_KINEMATIC_CHAIN
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')

class LitGenModel(LightningModule):
    def __init__(self, model, cfg, save_folder):
        super().__init__()

        # Model parameters
        self.cfg = cfg
        self.automatic_optimization = False
        self.save_folder = save_folder

        # Pytorch model
        self.model = model

        # Create save folder
        self.save_folder = os.path.join("results",save_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.normalizer = MotionNormalizer()

    def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []
        for i in range(2):
            joint = mp_data[:,i,:22*3].reshape(-1,22,3)
            mp_joint.append(joint)
        plot_3d_motion(result_path + "_skeleton.mp4", HML_KINEMATIC_CHAIN, mp_joint, title=caption, fps=30)

    def generate_one_sample(self, prompt_individual1, prompt_individual2, prompt_interaction, name):
        """
        Generate a single sample including the motion, the plots and save the influences
            :param prompt_individual1: Individual 1 prompt
            :param prompt_individual2: Individual 2 prompt
            :param prompt_interaction: Interaction prompt
            :param name: Name of the output file
        """

        # Set model to eval mode
        self.model.eval()

        batch = OrderedDict({})

        # Add dummy motion lens to the batch
        # This is required for the model to run but not important as we are not using masks on inference
        batch["motion_lens"] = torch.zeros(1,1).long().cuda()

        # Add prompts to the batch
        batch["prompt_individual1"] = prompt_individual1
        batch["prompt_individual2"] = prompt_individual2
        batch["prompt_interaction"] = prompt_interaction    

        # Generate motion sequence (window size seems to be fixed at 299)
        window_size = 299
        motion_o, influence1_h, influence2_h, out1_h, out2_h, out_influence_h = self.generate_loop(batch, window_size)

        result_path = f"{self.save_folder}/{name}"

        # Save motion output
        np.save(f"{result_path}_motion.npy", motion_o)

        # Save influences
        influence1_history = influence1_h
        influence1_history = np.array([influence1_history[i].cpu().detach().numpy() for i in range(len(influence1_history))])
        np.save(f"{result_path}_influence1.npy", influence1_history)

        influence2_history = influence2_h
        influence2_history = np.array([influence2_history[i].cpu().detach().numpy() for i in range(len(influence2_history))])
        np.save(f"{result_path}_influence2.npy", influence2_history)

        # Plot in the skeleton way
        self.plot_t2m(motion_o, result_path, batch["prompt_interaction"])

        """
        # Plot motion using aitvierwer
        plot_aitviewer(motion_o, self.save_folder, name)
        """
        
        # Plot influence histories
        plot_influence(influence1_history, influence2_history, self.model.mixing_mode, result_path)


    def generate_loop(self, batch, window_size):
        """
        Generate motion sequence for a given prompt
            :param batch: Input batch
            :param window_size: Window size for the motion sequence
        """

        # Extract prompts
        prompt_individual1 = batch["prompt_individual1"]
        prompt_individual2 = batch["prompt_individual2"]
        prompt_interaction = batch["prompt_interaction"]

        # Copy batch and set motion lens to window size
        batch = copy.deepcopy(batch)
        batch["motion_lens"][:] = window_size

        # Set batch prompts (no changes just using a different name)
        batch["text_individual1"] = [prompt_individual1]
        batch["text_individual2"] = [prompt_individual2]
        batch["text_interaction"] = [prompt_interaction]    

        # Run model
        batch = self.model(batch)

        # Extract motion output
        motion_output_both = batch["output"][0].reshape(batch["output"][0].shape[0], 2, -1)
        motion_output_both = motion_output_both.cpu().detach().numpy()
        #motion_output_both = self.normalizer.backward(motion_output_both)

        # Apply gaussian filter to the motion output
        motion_output_both = gaussian_filter1d(motion_output_both, 1, axis=0, mode='nearest')

        # Extract influences
        influence1_history = batch["influence_i1"]
        influence2_history = batch["influence_i2"]

        # Extract motion history
        out1_history = batch["out1"]
        out2_history = batch["out2"]
        out_influenced_history = batch["out_influenced"]
        

        return motion_output_both, influence1_history, influence2_history, out1_history, out2_history, out_influenced_history

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Argparse example with optional arguments")

    # Add optional arguments
    parser.add_argument('--model', type=str, required=True, help='Model Configuration file')
    parser.add_argument('--infer', type=str, required=True, help='Infer Configuration file')
    parser.add_argument('--device', type=str, required=True, help='Device to run the model')

    parser.add_argument('--text_individual1', type=str, required=True, help='Individual 1 prompt')
    parser.add_argument('--text_individual2', type=str, required=True, help='Individual 2 prompt')
    parser.add_argument('--text_interaction', type=str, required=True, help='Interaction prompt')

    parser.add_argument('--out', type=str, required=True, help='Folder to save the results')
    parser.add_argument('--name', type=str, required=True, help='Name of the output file')

    # Parse the arguments
    args = parser.parse_args()

    # Load config files and model checkpoint
    model_cfg = get_config(args.model)
    infer_cfg = get_config(args.infer)
    model = MixerMDM(model_cfg)

    # Load model checkpoint
    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt.keys()):
            if "model" in k:
                ckpt[k.replace("model.", "")] = ckpt.pop(k)
        model.load_state_dict(ckpt, strict=True)
        print("checkpoint state loaded!")

    # Lightning model wrapper for inference
    litmodel = LitGenModel(model, infer_cfg, args.out).to(torch.device("cuda:"+ args.device))

    # Generate X samples
    for i in range(10):
        litmodel.generate_one_sample(args.text_individual1, 
                                    args.text_individual2, 
                                    args.text_interaction, 
                                    args.name + f"_{i}")

