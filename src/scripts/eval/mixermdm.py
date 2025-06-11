import sys
sys.path.append(sys.path[0]+r"/../../")
import numpy as np
import torch
import os

from datetime import datetime
from evaluation.utils import EvaluatorModelWrapper, get_dataset_motion_loader_hml3d, get_motion_loader_humanml3d, EvaluatorModelWrapperIndividual
from evaluation.utils import get_dataset_motion_loader, get_motion_loader_in2IN
from models.mixermdm import MixerMDM
from utils.metrics import calculate_activation_statistics, calculate_diversity, calculate_frechet_distance, calculate_multimodality, calculate_top_k, euclidean_distance_matrix
from collections import OrderedDict
from utils.configs import get_config
from tqdm import tqdm
import argparse

def evaluate_matching_score(motion_loaders, eval_wrapper, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(motion_loader)):
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                mm_dist_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            mm_dist = mm_dist_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = mm_dist
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, eval_wrapper, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(groundtruth_loader)):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, eval_wrapper, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                batch[2] = batch[2][0]
                batch[3] = batch[3][0]
                batch[4] = batch[4][0]
                motion_embedings = eval_wrapper.get_motion_embeddings(batch)
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file, replication_times, eval_motion_loaders, gt_loader, eval_wrapper):
    with open(log_file, 'w') as f:

        all_metrics = OrderedDict({
            'MM Distance': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({}),
            'MultiModality': OrderedDict({}),
        })

        for replication in range(replication_times):

            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader

            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, eval_wrapper, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, eval_wrapper, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(mm_motion_loaders, eval_wrapper, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['MM Distance']:
                    all_metrics['MM Distance'][key] = [item]
                else:
                    all_metrics['MM Distance'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]

        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values))
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        
        return all_metrics


def calculate_f_score(log_file, metrics_ig, metrics_hml3d):
    with open(log_file, 'w') as f:
        print('========== F-Score Summary ==========')
        print('========== F-Score Summary ==========', file=f, flush=True)
        for metric_name in metrics_ig.keys():

            metric_dict_ig = metrics_ig[metric_name]
            metric_dict_hml3d = metrics_hml3d[metric_name]

            for (model_name_ig, values_ig), (model_name_hml3d, values_hml3d) in zip(metric_dict_ig.items(), metric_dict_hml3d.items()):
                mean_ig, conf_interval_ig = get_metric_statistics(np.array(values_ig))
                mean_hml3d, conf_interval_hml3d = get_metric_statistics(np.array(values_hml3d))

                if isinstance(mean_ig, np.float64) or isinstance(mean_ig, np.float32):
                    f_score = 2 * mean_ig * mean_hml3d / (mean_ig + mean_hml3d)
                    c_interval = (conf_interval_hml3d + conf_interval_ig) / 2
                    print(f'---> [{model_name_ig}][{metric_name}] F-Score: {f_score:.4f}, CInterval: {c_interval:.4f}')
                    print(f'---> [{model_name_ig}][{metric_name}] F-Score: {f_score:.4f}, CInterval: {c_interval:.4f}', file=f, flush=True)
                elif isinstance(mean_ig, np.ndarray):
                    line = f'---> [{model_name_ig}][{metric_name}]'
                    for i in range(len(mean_ig)):
                        f_score = 2 * mean_ig[i] * mean_hml3d[i] / (mean_ig[i] + mean_hml3d[i])
                        c_interval = (conf_interval_hml3d[i] + conf_interval_ig[i]) / 2
                        line += '(top %d) F-Score: %.4f CInt: %.4f;' % (i+1, f_score,c_interval)
                    print(line)
                    print(line, file=f, flush=True)

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Argparse example with optional arguments")

    # Add optional arguments
    parser.add_argument('--model', type=str, required=True, help='Model Configuration file')
    parser.add_argument('--name', type=str, required=True, help='Model Configuration file')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--align', type=bool, default=True, help='Align the motions')
    parser.add_argument('--llm' , type=bool, default=False, help='Use the LLM model')

    # Parse the arguments
    args = parser.parse_args()
    mm_num_samples = 50
    mm_num_repeats = 15
    mm_num_times = 5
    diversity_times = 125
    replication_times = 2
    num_samples = 250

    print(f"Align: {args.align}")

    # Device
    device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    # Datasets
    data_cfg_hml3d = get_config("configs/datasets.yaml").humanml3d_test
    data_cfg_ig = get_config("configs/datasets.yaml").interhuman_test

    # Get folder
    model_name = args.name

    # Create an evaluation folder for the results
    os.makedirs('evaluation_logs', exist_ok=True)
    os.makedirs(os.path.join('evaluation_logs',model_name), exist_ok=True)

    
    # Output folder for the evaluation
    output_folder = os.path.join('evaluation_logs', model_name)
    print(f"Evaluating model {model_name}")

    # Model
    model_cfg = get_config(args.model)
    model = MixerMDM(model_cfg, align=args.align) 

    """
    # Loading the checkpoint
    checkpoint = torch.load(model_cfg.CHECKPOINT, map_location=torch.device("cpu")) 
    for k in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][k[6:]] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    """

    # Loading the checkpoint
    checkpoint = torch.load(model_cfg.CHECKPOINT, map_location=torch.device("cpu")) 
    for k in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][k[6:]] = checkpoint["state_dict"].pop(k)
    for k in list(checkpoint["state_dict"].keys()):
        if "model1" in k or "model2" in k or "denoiser1" in k or "denoiser2" in k:
            checkpoint["state_dict"].pop(k)
    print("Final Swaped Keys:")
    for k in list(checkpoint["state_dict"].keys()):
        print(k)
    model.load_state_dict(checkpoint['state_dict'], strict=False)


    # Change the mode of the mixing models
    model.mixing.mode = "eval" 

    # Evaluation motion loaders
    eval_motion_loaders_ig = {}
    gt_loader_ig, gt_dataset_ig = get_dataset_motion_loader(data_cfg_ig, 96, num_samples)
    eval_motion_loaders_ig[model_cfg.NAME] = lambda: get_motion_loader_in2IN(
        96,
        model,
        gt_dataset_ig,
        device,
        mm_num_samples,
        mm_num_repeats,
        llm=args.llm,
        normalize=False
    )

    eval_motion_loaders_hml3d = {}
    gt_loader_hml3d, gt_dataset_hml3d = get_dataset_motion_loader_hml3d(data_cfg_hml3d, 32, num_samples)
    eval_motion_loaders_hml3d[model_cfg.NAME] = lambda: get_motion_loader_humanml3d(
        32,
        model,
        gt_dataset_hml3d,
        device,
        mm_num_samples,
        mm_num_repeats,
        normalize=False
    )

    # Evaluator model
    evalmodel_cfg_hml3d = get_config("configs/eval_individual.yaml")
    eval_wrapper_hml3d = EvaluatorModelWrapperIndividual(evalmodel_cfg_hml3d, device)
    evalmodel_cfg_ig = get_config("configs/eval.yaml")
    eval_wrapper_ig = EvaluatorModelWrapper(evalmodel_cfg_ig, device)

    # Evaluation of the interaction
    metrics_ig = evaluation(
        log_file=os.path.join(output_folder, 'ih.txt'),
        replication_times=replication_times,
        eval_motion_loaders=eval_motion_loaders_ig,
        gt_loader=gt_loader_ig,
        eval_wrapper=eval_wrapper_ig
    )

    # Evaluation of the humanml3d
    metrics_hml3d = evaluation(
        log_file=os.path.join(output_folder, 'hml3d.txt'),
        replication_times=replication_times,
        eval_motion_loaders=eval_motion_loaders_hml3d,
        gt_loader=gt_loader_hml3d,
        eval_wrapper=eval_wrapper_hml3d
    )

    # Calculate the armonic mean
    calculate_f_score(
        log_file=os.path.join(output_folder, 'f_score.txt'),
        metrics_ig=metrics_ig,
        metrics_hml3d=metrics_hml3d
    )

