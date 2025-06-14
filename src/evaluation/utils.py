import copy
from datasets.humanml3d import HumanML3D
import numpy as np
import torch

from evaluation.datasets import EvaluationDatasetDualMDM, EvaluationDatasetHumanML3D, EvaluationDatasetInterHuman, MMGeneratedDatasetHumanML3D, MMGeneratedDatasetInterHuman
from models import *
from datasets import InterHuman
from evaluation.models import InterCLIP
from torch.utils.data import DataLoader
from utils.alignment import center_motion, ih_to_smpl, smpl_to_ih

def get_dataset_motion_loader(opt, batch_size, num_samples=-1):
    """
    Get the ground truth dataset of motions with his given dataloader.
        :param opt: Configuration of the dataset.
        :param batch_size: Batch size of the dataloader.
        :return: Dataloader of the motion datase.
    """
    opt = copy.deepcopy(opt)
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = InterHuman(opt, num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset


def get_dataset_motion_loader_hml3d(opt, batch_size, num_samples=-1):
    """
    Get the ground truth dataset of motions with his given dataloader.
        :param opt: Configuration of the dataset.
        :param batch_size: Batch size of the dataloader.
        :return: Dataloader of the motion datase.
    """
    dataset = HumanML3D(opt, extended=True, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset


def get_motion_loader_in2IN(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, llm, normalize):
    """
    Get the generated dataset of motions with his given dataloader and the MultiModality one.
        :param batch_size: Batch size of the dataloader.
        :param model: Model to generate the motions.
        :param ground_truth_dataset: Ground truth dataset.
        :param device: Device to run the model.
        :param mm_num_samples: Number of samples to generate for the MultiModality metric.
        :param mm_num_repeats: Number of repeats for each sample in the MultiModality metric.
        :return: Dataloader of the generated motion dataset and the MultiModality one.
    """

    dataset = EvaluationDatasetInterHuman(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, llm=llm, normalize=normalize)
    mm_dataset = MMGeneratedDatasetInterHuman(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader

def get_motion_loader_humanml3d(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, normalize):
    """
    Get the generated dataset of motions with his given dataloader and the MultiModality one.
        :param batch_size: Batch size of the dataloader.
        :param model: Model to generate the motions.
        :param ground_truth_dataset: Ground truth dataset.
        :param device: Device to run the model.
        :param mm_num_samples: Number of samples to generate for the MultiModality metric.
        :param mm_num_repeats: Number of repeats for each sample in the MultiModality metric.
        :return: Dataloader of the generated motion dataset and the MultiModality one.
    """

    dataset = EvaluationDatasetHumanML3D(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, normalize=normalize)
    mm_dataset = MMGeneratedDatasetHumanML3D(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader



def get_motion_loader_DualMDM(batch_size, model, ground_truth_dataset, device, num_repeats, normalize):
    """
    Get the generated dataset of motions with his given dataloader
        :param batch_size: Batch size of the dataloader.
        :param model: Model to generate the motions.
        :param ground_truth_dataset: Ground truth dataset.
        :param device: Device to run the model.
        :param num_repeats: Number of repeats for each sample.
        :return: Dataloader of the generated motion dataset.
    """
    dataset = EvaluationDatasetDualMDM(model, ground_truth_dataset, device, num_repeats=num_repeats, normalize=normalize)
    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    return motion_loader


def build_models(cfg):
    """
    Create and load the feature extractor model for the evaluation.
        :param cfg: Configuration of the model.
        :return: Feature extractor model for the evaluation.
    """

    model = InterCLIP(cfg)

    # Load the model from the checkpoint
    checkpoint = torch.load(cfg.CHECKPOINT, map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model


class EvaluatorModelWrapper(object):
    """
    Wrapper of the model for the evaluation.
    The model will be used to extract features from the generated motions and the gt motions.
    """
    def __init__(self, cfg, device):
        """
        Initialization of the model.
            :param cfg: Configuration of the model.
            :param device: Device to run the model.
        """
        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        self.extended = cfg.EXTENDED


    def get_co_embeddings(self, batch_data):
        """
        Get the embeddings of the text and the motions of a given batch of data.
            :param batch_data: Batch of data to extract the embeddings.
            :return: Embeddings of the text and the motions.
        Please note that the results does not following the order of inputs
        """
        with torch.no_grad():
            # Extract data from the batch provided by the evaluation datasets 
            if self.extended:
                name, text, motion1, motion2, motion_lens, text_individual1, text_individual2 = batch_data
            else:
                name, text, motion1, motion2, motion_lens = batch_data
            
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)
            if self.extended:
                text_individual1 = list(text_individual1)
                text_individual2 = list(text_individual2)

            # Create padding for the motions
            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            # Create batch for feature prediction
            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens
            if self.extended:
                batch["text_individual1"] = text_individual1
                batch["text_individual2"] = text_individual2

            # Motion Encoding
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

            # Text Encoding
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    def get_motion_embeddings(self, batch_data):
        """
        Get the embeddings of the motions of a given batch of data.
            :param batch_data: Batch of data to extract the embeddings.
            :return: Embeddings of the motions.
        Please note that the results does not following the order of inputs
        """
        with torch.no_grad():
            # Extract data from the batch provided by the evaluation datasets
            if self.extended:
                name, text, motion1, motion2, motion_lens, text_individual1, text_individual2 = batch_data
            else:
                name, text, motion1, motion2, motion_lens = batch_data
            
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            # Create padding for the motions
            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            # Create batch for feature prediction
            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens
            if self.extended:
                batch["text_individual1"] = text_individual1
                batch["text_individual2"] = text_individual2

            # Motion Encoding
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding

class EvaluatorModelWrapperIndividual(object):
    """
    Wrapper of the model for the evaluation.
    The model will be used to extract features from the generated motions and the gt motions.
    """
    def __init__(self, cfg, device):
        """
        Initialization of the model.
            :param cfg: Configuration of the model.
            :param device: Device to run the model.
        """
        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        self.extended = cfg.EXTENDED


    def get_co_embeddings(self, batch_data):
        """
        Get the embeddings of the text and the motions of a given batch of data.
            :param batch_data: Batch of data to extract the embeddings.
            :return: Embeddings of the text and the motions.
        Please note that the results does not following the order of inputs
        """
        with torch.no_grad():
            # Extract data from the batch provided by the evaluation datasets 
            if self.extended:
                name, text, motion1, motion2, motion_lens, text_individual1, text_individual2 = batch_data
            else:
                name, text, motion1, motion2, motion_lens = batch_data
            
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()


            #  Extract text from the batch and sort it by length
            text = list(text)
            text_individual1 = list(text_individual1)
            text_individual2 = list(text_individual2)

            # Create a new text list alternating one from each individual
            text = [text_individual1[i//2] if i % 2 == 0 else text_individual2[i//2] for i in range(len(text)*2)]

            # Center the motions for fair comparison
            motion1 = motions[..., :motion1.shape[-1]]
            motion2 = motions[..., motion1.shape[-1]:]
            motion1 = smpl_to_ih(center_motion(ih_to_smpl(motion1)))
            motion2 = smpl_to_ih(center_motion(ih_to_smpl(motion2)))

            # Interleave motions at the batch level
            # First, expand the batch dimension to separate each sequence
            motion1_expanded = motion1.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, feature_dim)
            motion2_expanded = motion2.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, feature_dim)

            # Concatenate along the new dimension to interleave them
            interleaved_motions = torch.cat([motion1_expanded, motion2_expanded], dim=1)  # Shape: (batch_size, 2, seq_len, feature_dim)

            # Reshape to alternate motions at the batch level
            interleaved_motions = interleaved_motions.view(motion1.shape[0] * 2, motion1.shape[1], -1)
            motions = interleaved_motions.detach().to(self.device).float()
            motion_lens = motion_lens.repeat_interleave(2)

            # Create padding for the motions
            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            # Create batch for feature prediction
            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            # Motion Encoding
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

            # Text Encoding
            text_embedding = self.model.encode_text(batch)['text_emb']
            text_embedding = text_embedding

        return text_embedding, motion_embedding

    def get_motion_embeddings(self, batch_data):
        """
        Get the embeddings of the motions of a given batch of data.
            :param batch_data: Batch of data to extract the embeddings.
            :return: Embeddings of the motions.
        Please note that the results does not following the order of inputs
        """
        with torch.no_grad():
            # Extract data from the batch provided by the evaluation datasets 
            if self.extended:
                name, text, motion1, motion2, motion_lens, text_individual1, text_individual2 = batch_data
            else:
                name, text, motion1, motion2, motion_lens = batch_data
            
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            #  Extract text from the batch and sort it by length
            text = list(text)
            text_individual1 = list(text_individual1)
            text_individual2 = list(text_individual2)

            # Create a new text list alternating one from each individual
            text = [text_individual1[i//2] if i % 2 == 0 else text_individual2[i//2] for i in range(len(text)*2)]

            # Center the motions for fair comparison
            motion1 = motions[..., :motion1.shape[-1]]
            motion2 = motions[..., motion1.shape[-1]:]
            motion1 = smpl_to_ih(center_motion(ih_to_smpl(motion1)))
            motion2 = smpl_to_ih(center_motion(ih_to_smpl(motion2)))

            # Interleave motions at the batch level
            # First, expand the batch dimension to separate each sequence
            motion1_expanded = motion1.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, feature_dim)
            motion2_expanded = motion2.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, feature_dim)

            # Concatenate along the new dimension to interleave them
            interleaved_motions = torch.cat([motion1_expanded, motion2_expanded], dim=1)  # Shape: (batch_size, 2, seq_len, feature_dim)

            # Reshape to alternate motions at the batch level
            interleaved_motions = interleaved_motions.view(motion1.shape[0] * 2, motion1.shape[1], -1)
            motions = interleaved_motions.detach().to(self.device).float()
            motion_lens = motion_lens.repeat_interleave(2)

            # Create padding for the motions
            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            # Create batch for feature prediction
            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens
            
            # Motion Encoding
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding