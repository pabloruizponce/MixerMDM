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
- [2025-04-2] Our paper is available on [arXiv](https://arxiv.org/abs/2504.01019)
- [2025-02-26] MixerMDM is accepted at CVPR 2025!

## 📝 TODO List
- [ ] Release code
- [ ] Release model weights
- [ ] Release visualization code.


## 📚 Citation

If you find our work helpful, please cite:

```bibtex
@misc{ruizponce2025mixermdmlearnablecompositionhuman,
      title={MixerMDM: Learnable Composition of Human Motion Diffusion Models}, 
      author={Pablo Ruiz-Ponce and German Barquero and Cristina Palmero and Sergio Escalera and José García-Rodríguez},
      year={2025},
      eprint={2504.01019},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.01019}, 
}
```
