<h1 align="center">MixerMDM: Learnable Composition of Human Motion Diffusion Models</h1>

  <p align="center">
    <a href="https://www.pabloruizponce.com/papers/MixerMDM"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a>
    <a href="https://arxiv.org/abs/2504.01019"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2504.01019-b31b1b.svg"></a> 
  </p>
  
<br>

## 🔎 About
<div style="text-align: center;">
    <img src="https://pycelle.com/papers/MixerMDM/cover.png" align="center" width=100% >
</div>
</br>
Generating human motion guided by conditions such as textual descriptions is challenging due to the need for datasets with pairs of high-quality motion and their corresponding conditions. The difficulty increases when aiming for finer control in the generation. To that end, prior works have proposed to combine several motion diffusion models pre-trained on datasets with different types of conditions, thus allowing control with multiple conditions. However, the proposed merging strategies overlook that the optimal way to combine the generation processes might depend on the particularities of each pre-trained generative model and also the specific textual descriptions. In this context, we introduce MixerMDM, the first learnable model composition technique for combining pre-trained text-conditioned human motion diffusion models. Unlike previous approaches, MixerMDM provides a dynamic mixing strategy that is trained in an adversarial fashion to learn to combine the denoising process of each model depending on the set of conditions driving the generation. By using MixerMDM to combine single- and multi-person motion diffusion models, we achieve fine-grained control on the dynamics of every person individually, and also on the overall interaction. Furthermore, we propose a new evaluation technique that, for the first time in this task, measures the interaction and individual quality by computing the alignment between the mixed generated motions and their conditions as well as the capabilities of MixerMDM to adapt the mixing throughout the denoising process depending on the motions to mix.

## 📌 News
- Code and model weights are now available!
- Our paper is available on [arXiv](https://arxiv.org/abs/2504.01019)
- MixerMDM is accepted at CVPR 2025!

## 📝 TODO List
- [x] Release code
- [x] Release model weights
- [ ] Release visualization code.

## 💻 Usage
### 🛠️ Installation
1. Clone the repo
  ```sh
  git clone https://github.com/pabloruizponce/MixerMDM.git
  ```
2. Install the requirements
   1. Install conda environment
      ```sh
      conda env create -f environment.yaml
      ```
   2. Activate the conda environment
      ```sh
      conda activate MixerMDM
      ```

### 🕹️ Inference
Download the model weights from [here](https://drive.google.com/drive/folders/1yqOYckYmuksSySLOf-qETJzzbinFa7tY?usp=share_link) and place them in the `checkpoints` folder.

```sh
  python src/scripts/infer/mixermdm.py \
      --model configs/models/MixerMDM.yaml \
      --infer configs/infer.yaml \
      --out results \
      --device 0 \
      --text_interaction "Interaction textual description" \
      --text_individual1 "Individual textual description" \
      --text_individual2 "Individual textual description" \
      --name "output_name" \
```

> [!NOTE]  
> More information about the parameters can be found using the `--help` flag.

## 📚 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{ruiz2025mixermdm,
  title={Mixermdm: Learnable composition of human motion diffusion models},
  author={Ruiz-Ponce, Pablo and Barquero, German and Palmero, Cristina and Escalera, Sergio and Garc{\'\i}a-Rodr{\'\i}guez, Jos{\'e}},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={12380--12390},
  year={2025}
}
```
