# Simulating the Real World: Survey & Resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FALEEEHU%2FAwesome-Text2X-Resources%2F&count_bg=%23EAA8EA&title_bg=%233D2549&icon=react.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-pink.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-pink) ![Stars](https://img.shields.io/github/stars/ALEEEHU/Awesome-Text2X-Resources)

This repository is divided into two main sections:

> **Our Survey Paper Collection** - This section presents our survey, _"Simulating the Real World: A Unified Survey of Multimodal Generative Models"_, which systematically unify the study of 2D, video, 3D and 4D generation within a single framework.

> **Text2X Resources** – This section continues the original Awesome-Text2X-Resources, an open collection of state-of-the-art (SOTA) and novel Text-to-X (X can be everything) methods, including papers, codes, and datasets. The goal is to track the rapid progress in this field and provide researchers with up-to-date references.

⭐ If you find this repository useful for your research or work, a star is highly appreciated!

💗 This repository is continuously updated. If you find relevant papers, blog posts, videos, or other resources that should be included, feel free to submit a pull request (PR) or open an issue. Community contributions are always welcome!

<img src="./media/add_oil.png" width=15% align="right" />

## Table of Contents
- [Our Survey Paper Collection](#-our-survey-paper-collection)
	- [Abstract](#abstract)
  - [Paradigms](#paradigms)
    * [2D Generation](#2d-generation)
    * [Video Generation](#video-generation)
      * [Algorithms](#video-algorithms)
      * [Applications](#video-applications)
    * [3D Generation](#3d-generation)
      * [Algorithms](#3d-algorithms)
      * [Applications](#3d-applications)
    * [4D Generation](#4d-generation)
      * [Algorithms](#4d-algorithms)
      * [Applications](#4d-applications)
- [Text2X Resources](#-awesome-text2x-resources)
	- [Text to 4D](#text-to-4d)
	  * [ArXiv Papers](#-4d-arxiv-papers)
	- [Text to Video](#text-to-video)
	  * [ArXiv Papers](#-t2v-arxiv-papers)
	  * [Additional Info](#video-other-additional-info)
	- [Text to 3D Scene](#text-to-scene)
	  * [ArXiv Papers](#-3d-scene-arxiv-papers)
	- [Text to Human Motion](#text-to-human-motion)
	  * [ArXiv Papers](#-motion-arxiv-papers)
	  * [Additional Info](#motion-other-additional-info)
	- [Text to 3D Human](#text-to-3d-human)
	  * [ArXiv Papers](#-human-arxiv-papers)
	- [Related Resources](#related-resources)
	  * [Text to Other Tasks](#text-to-other-tasks)
	  * [Survey and Awesome Repos](#survey-and-awesome-repos)

## 📜 Our Survey Paper Collection
<p align=center> 𝐒𝐢𝐦𝐮𝐥𝐚𝐭𝐢𝐧𝐠 𝐭𝐡𝐞 𝐑𝐞𝐚𝐥 𝐖𝐨𝐫𝐥𝐝: 𝐀 𝐔𝐧𝐢𝐟𝐢𝐞𝐝 𝐒𝐮𝐫𝐯𝐞𝐲 𝐨𝐟 𝐌𝐮𝐥𝐭𝐢𝐦𝐨𝐝𝐚𝐥 𝐆𝐞𝐧𝐞𝐫𝐚𝐭𝐢𝐯𝐞 𝐌𝐨𝐝𝐞𝐥𝐬 </p>

<div align=center>

[![arXiv](https://img.shields.io/badge/arXiv-2503.04641-b31b1b.svg)](https://arxiv.org/abs/2503.04641)

</div>

<p align="center"> <img src="./media/teaser.png" width="90%" height="90%"> </p>

> ### Abstract
Understanding and replicating the real world is a critical challenge in Artificial General Intelligence (AGI) research. To achieve this, many existing approaches, such as world models, aim to capture the fundamental principles governing the physical world, enabling more accurate simulations and meaningful interactions. However, current methods often treat different modalities, including 2D (images), videos, 3D, and 4D representations, as independent domains, overlooking their interdependencies. Additionally, these methods typically focus on isolated dimensions of reality without systematically integrating their connections. In this survey, we present a unified survey for multimodal generative models that investigate the progression of data dimensionality in real-world simulation. Specifically, this survey starts from 2D generation (appearance), then moves to video (appearance+dynamics) and 3D generation (appearance+geometry), and finally culminates in 4D generation that integrate all dimensions. To the best of our knowledge, this is the first attempt to systematically unify the study of 2D, video, 3D and 4D generation within a single framework. To guide future research, we provide a comprehensive review of datasets, evaluation metrics and future directions, and fostering insights for newcomers. This survey serves as a bridge to advance the study of multimodal generative models and real-world simulation within a unified framework.

> ### ⭐ Citation

If you find this paper and repo helpful for your research, please cite it below:

```bibtex

@article{hu2025simulatingrealworldunified,
  title={Simulating the Real World: A Unified Survey of Multimodal Generative Models},
  author={Yuqi Hu and Longguang Wang and Xian Liu and Ling-Hao Chen and Yuwei Guo and Yukai Shi and Ce Liu and Anyi Rao and Zeyu Wang and Hui Xiong},
  journal={arXiv preprint arXiv:2503.04641},
  year={2025}
}

```

## Paradigms

> [!TIP]
> *Feel free to pull requests or contact us if you find any related papers that are not included here.* The process to submit a pull request is as follows:
- a. Fork the project into your own repository.
- b. Add the Title, Paper link, Conference, Project/GitHub link in `README.md` using the following format:
 ```
[Origin] **Paper Title** [[Paper](Paper Link)] [[GitHub](GitHub Link)] [[Project Page](Project Page Link)]
 ```
- c. Submit the pull request to this branch.

### 2D Generation

##### Text-to-Image Generation.
Here are some seminal papers and models.

* **Imagen**: [NeurIPS 2022] **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding** [[Paper](https://arxiv.org/abs/2205.11487)] [[Project Page](https://imagen.research.google/)]
* **DALL-E**: [ICML 2021] **Zero-shot text-to-image generation** [[Paper](https://arxiv.org/abs/2102.12092)] [[GitHub](https://github.com/openai/DALL-E)]
* **DALL-E 2**: [arXiv 2022] **Hierarchical Text-Conditional Image Generation with CLIP Latents** [[Paper](https://arxiv.org/abs/2204.06125)]
* **DALL-E 3**: [[Platform Link](https://openai.com/index/dall-e-3/)]
* **DeepFloyd IF**: [[GitHub](https://github.com/deep-floyd/IF)]
* **Stable Diffusion**: [CVPR 2022] **High-Resolution Image Synthesis with Latent Diffusion Models** [[Paper](https://arxiv.org/abs/2112.10752)] [[GitHub](https://github.com/CompVis/latent-diffusion)]
* **SDXL**: [ICLR 2024 spotlight] **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis** [[Paper](https://arxiv.org/abs/2307.01952)] [[GitHub](https://github.com/Stability-AI/generative-models)]
* **FLUX.1**: [[Platform Link](https://blackforestlabs.ai/)]

----


### Video Generation
Text-to-video generation models adapt text-to-image frameworks to handle the additional dimension of dynamics in the real world. We classify these models into _three_ categories based on different generative machine learning architectures.

> ##### Survey
* [AIRC 2023] **A Survey of AI Text-to-Image and AI Text-to-Video Generators** [[Paper](https://arxiv.org/abs/2311.06329)] 
* [arXiv 2024] **Sora as an AGI World Model? A Complete Survey on Text-to-Video Generation** [[Paper](https://arxiv.org/abs/2403.05131)]

#### Video Algorithms

> ##### (1) VAE- and GAN-based Approaches.
VAE-based Approaches.
* **SV2P**: [ICLR 2018 Poster] **Stochastic Variational Video Prediction** [[Paper](https://arxiv.org/abs/1710.11252)] [[Project Page](https://sites.google.com/site/stochasticvideoprediction/)]
* [arXiv 2021] **FitVid: Overfitting in Pixel-Level Video Prediction** [[Paper](https://arxiv.org/abs/2403.05131)] [[GitHub](https://github.com/google-research/fitvid)] [[Project Page](https://sites.google.com/view/fitvidpaper)]

GAN-based Approaches.
* [CVPR 2018] **MoCoGAN: Decomposing Motion and Content for Video Generation** [[Paper](https://arxiv.org/abs/1707.04993)] [[GitHub](https://github.com/sergeytulyakov/mocogan)] 
* [CVPR 2022] **StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2** [[Paper](https://arxiv.org/abs/2112.14683)] [[GitHub](https://github.com/universome/stylegan-v)] [[Project Page](https://skor.sh/stylegan-v)]
* **DIGAN**: [ICLR 2022] **Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks** [[Paper](https://arxiv.org/abs/2202.10571)] [[GitHub](https://github.com/sihyun-yu/digan)] [[Project Page](https://sihyun.me/digan/)]
* [ICCV 2023] **StyleInV: A Temporal Style Modulated Inversion Network for Unconditional Video Generation** [[Paper](https://arxiv.org/abs/2308.16909)] [[GitHub](https://github.com/johannwyh/StyleInV)] [[Project Page](https://www.mmlab-ntu.com/project/styleinv/index.html)]

> ##### (2) Diffusion-based Approaches.
U-Net-based Architectures.
* [NeurIPS 2022] **Video Diffusion Models** [[Paper](https://arxiv.org/abs/2204.03458)] [[Project Page](https://video-diffusion.github.io/)] 
* [arXiv 2022] **Imagen Video: High Definition Video Generation with Diffusion Models** [[Paper](https://arxiv.org/abs/2210.02303)] [[Project Page](https://imagen.research.google/video/)]
* [arXiv 2022] **MagicVideo: Efficient Video Generation With Latent Diffusion Models** [[Paper](https://arxiv.org/abs/2211.11018)] [[Project Page](https://magicvideo.github.io/#)]
* [ICLR 2023 Poster] **Make-A-Video: Text-to-Video Generation without Text-Video Data** [[Paper](https://arxiv.org/abs/2209.14792)] [[Project Page](https://make-a-video.github.io/)]
* **GEN-1**: [ICCV 2023] **Structure and Content-Guided Video Synthesis with Diffusion Models** [[Paper](https://arxiv.org/abs/2302.03011)] [[Project Page](https://runwayml.com/research/gen-1)]
* **PYoCo**: [ICCV 2023] **Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models** [[Paper](https://arxiv.org/abs/2305.10474)] [[Project Page](https://research.nvidia.com/labs/dir/pyoco/)]
* [CVPR 2023] **Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models** [[Paper](https://arxiv.org/abs/2304.08818)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)]
* [IJCV 2024] **Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation** [[Paper](https://arxiv.org/abs/2309.15818)] [[GitHub](https://github.com/showlab/Show-1)] [[Project Page](https://showlab.github.io/Show-1/)]
* [NeurIPS 2024] **VideoComposer: Compositional Video Synthesis with Motion Controllability** [[Paper](https://arxiv.org/abs/2306.02018)] [[GitHub](https://github.com/ali-vilab/videocomposer)] [[Project Page](https://videocomposer.github.io/)] 
* [ICLR 2024 Spotlight] **AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning** [[Paper](https://arxiv.org/abs/2307.04725)] [[GitHub](https://github.com/guoyww/AnimateDiff)] [[Project Page](https://animatediff.github.io/)] 
* [CVPR 2024] **Make Pixels Dance: High-Dynamic Video Generation** [[Paper](https://arxiv.org/abs/2311.10982)] [[Project Page](https://makepixelsdance.github.io/)]
* [ECCV 2024] **Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning** [[Paper](https://arxiv.org/abs/2311.10709)] [[Project Page](https://emu-video.metademolab.com/)]
* [SIGGRAPH Asia 2024] **Lumiere: A Space-Time Diffusion Model for Video Generation** [[Paper](https://arxiv.org/abs/2401.12945)] [[Project Page](https://lumiere-video.github.io/)]

Transformer-based Architectures.
* [ICLR 2024 Poster] **VDT: General-purpose Video Diffusion Transformers via Mask Modeling** [[Paper](https://arxiv.org/abs/2305.13311)] [[GitHub](https://github.com/RERV/VDT)] [[Project Page](https://vdt-2023.github.io/)]
* **W.A.L.T**: [ECCV 2024] **Photorealistic Video Generation with Diffusion Models** [[Paper](https://arxiv.org/abs/2312.06662)] [[Project Page](https://walt-video-diffusion.github.io/)]
* [CVPR 2024] **Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis** [[Paper](https://arxiv.org/abs/2402.14797)] [[Project Page](https://snap-research.github.io/snapvideo/)]
* [CVPR 2024] **GenTron: Diffusion Transformers for Image and Video Generation** [[Paper](https://arxiv.org/abs/2312.04557)] [[Project Page](https://www.shoufachen.com/gentron_website/)]
* [ICLR 2025 Poster] **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer** [[Paper](https://arxiv.org/abs/2408.06072)] [[GitHub](https://github.com/THUDM/CogVideo)]
* [ICLR 2025 Spotlight] **Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers** [[Paper](https://arxiv.org/abs/2405.05945)] [[GitHub](https://github.com/Alpha-VLLM/Lumina-T2X)]

> ##### (3) Autoregressive-based Approaches.
* **VQ-GAN**: [CVPR 2021 Oral] **Taming Transformers for High-Resolution Image Synthesis** [[Paper](https://arxiv.org/abs/2012.09841)] [[GitHub](https://github.com/CompVis/taming-transformers)] 
* [CVPR 2023 Highlight] **MAGVIT: Masked Generative Video Transformer** [[Paper](https://arxiv.org/abs/2212.05199)] [[GitHub](https://github.com/google-research/magvit)] [[Project Page](https://magvit.cs.cmu.edu/)]
* [ICLR 2023 Poster] **CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers** [[Paper](https://arxiv.org/abs/2205.15868)] [[GitHub](https://github.com/THUDM/CogVideo)]
* [ICML 2024] **VideoPoet: A Large Language Model for Zero-Shot Video Generation** [[Paper](https://arxiv.org/abs/2312.14125)] [[Project Page](https://sites.research.google/videopoet/)]
* [ICLR 2024 Poster] **Language Model Beats Diffusion - Tokenizer is key to visual generation** [[Paper](https://arxiv.org/abs/2310.05737)] 
* [arXiv 2024] **Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation** [[Paper](https://arxiv.org/abs/2409.04410)] [[GitHub](https://github.com/TencentARC/SEED-Voken)]
* [arXiv 2024] **Emu3: Next-Token Prediction is All You Need** [[Paper](https://arxiv.org/abs/2409.18869)] [[GitHub](https://github.com/baaivision/Emu3)] [[Project Page](https://emu.baai.ac.cn/about)]
* [ICLR 2025 Poster] **Accelerating Auto-regressive Text-to-Image Generation with Training-free Speculative Jacobi Decoding** [[Paper](https://arxiv.org/abs/2410.01699)] [[GitHub](https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD/)]

#### Video Applications
> ##### Video Editing.
* [ICCV 2023] **Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation** [[Paper](https://arxiv.org/abs/2212.11565)] [[GitHub](https://github.com/showlab/Tune-A-Video)] [[Project Page](https://tuneavideo.github.io/)]
* [ICCV 2023] **Pix2Video: Video Editing using Image Diffusion** [[Paper](https://arxiv.org/abs/2303.12688)] [[GitHub](https://github.com/duyguceylan/pix2video)] [[Project Page](https://duyguceylan.github.io/pix2video.github.io/)]
* [CVPR 2024] **VidToMe: Video Token Merging for Zero-Shot Video Editing** [[Paper](https://arxiv.org/abs/2312.10656)] [[GitHub](https://github.com/lixirui142/VidToMe)] [[Project Page](https://vidtome-diffusion.github.io/)]
* [CVPR 2024] **Video-P2P: Video Editing with Cross-attention Control** [[Paper](https://arxiv.org/abs/2303.04761)] [[GitHub](https://github.com/dvlab-research/Video-P2P)] [[Project Page](https://video-p2p.github.io/)]
* [CVPR 2024 Highlight] **CoDeF: Content Deformation Fields for Temporally Consistent Video Processing** [[Paper](https://arxiv.org/abs/2308.07926)] [[GitHub](https://github.com/ant-research/CoDeF)] [[Project Page](https://qiuyu96.github.io/CoDeF/)]
* [NeurIPS 2024] **Towards Consistent Video Editing with Text-to-Image Diffusion Models** [[Paper](https://arxiv.org/abs/2305.17431)]
* [ICLR 2024 Poster] **Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models** [[Paper](https://arxiv.org/abs/2310.01107)] [[GitHub](https://github.com/Ground-A-Video/Ground-A-Video)] [[Project Page](https://ground-a-video.github.io/)]
* [arXiv 2024] **UniEdit: A Unified Tuning-Free Framework for Video Motion and Appearance Editing** [[Paper](https://arxiv.org/abs/2402.13185)] [[GitHub](https://github.com/JianhongBai/UniEdit)] [[Project Page](https://jianhongbai.github.io/UniEdit/)]
* [TMLR 2024] **AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks** [[Paper](https://arxiv.org/abs/2403.14468)] [[GitHub](https://github.com/TIGER-AI-Lab/AnyV2V)] [[Project Page](https://tiger-ai-lab.github.io/AnyV2V/)]

> ##### Novel View Synthesis.
* [arXiv 2024] **ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis** [[Paper](https://arxiv.org/abs/2409.02048)] [[GitHub](https://github.com/Drexubery/ViewCrafter)] [[Project Page](https://drexubery.github.io/ViewCrafter/)]
* [CVPR 2024 Highlight] **ViVid-1-to-3: Novel View Synthesis with Video Diffusion Models** [[Paper](https://arxiv.org/abs/2312.01305)] [[GitHub](https://github.com/ubc-vision/vivid123)] [[Project Page](https://jgkwak95.github.io/ViVid-1-to-3/)]
* [ICLR 2025 Poster] **CameraCtrl: Enabling Camera Control for Video Diffusion Models** [[Paper](https://arxiv.org/abs/2404.02101)] [[GitHub](https://github.com/hehao13/CameraCtrl)] [[Project Page](https://hehao13.github.io/projects-CameraCtrl/)]
* [ICLR 2025 Poster] **NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer** [[Paper](https://arxiv.org/abs/2405.15364)] [[GitHub](https://github.com/ZHU-Zhiyu/NVS_Solver)] 

> ##### Human Animation in Videos.
* [ICCV 2019] **Everybody Dance Now** [[Paper](https://arxiv.org/abs/1808.07371)] [[GitHub](https://github.com/carolineec/EverybodyDanceNow)] [[Project Page](https://carolineec.github.io/everybody_dance_now/)]
* [ICCV 2019] **Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis** [[Paper](https://arxiv.org/abs/1909.12224)] [[GitHub](https://github.com/svip-lab/impersonator)] [[Project Page](https://svip-lab.github.io/project/impersonator.html)] [[Dataset](https://svip-lab.github.io/dataset/iPER_dataset.html)]
* [NeurIPS 2019] **First Order Motion Model for Image Animation** [[Paper](https://arxiv.org/abs/2003.00196)] [[GitHub](https://github.com/AliaksandrSiarohin/first-order-model)] [[Project Page](https://aliaksandrsiarohin.github.io/first-order-model-website/)]
* [ICCV 2023] **Adding Conditional Control to Text-to-Image Diffusion Models** [[Paper](https://arxiv.org/abs/2302.05543)] [[GitHub](https://github.com/lllyasviel/ControlNet)]
* [ICCV 2023] **HumanSD: A Native Skeleton-Guided Diffusion Model for Human Image Generation** [[Paper](https://arxiv.org/abs/2304.04269)] [[GitHub](https://github.com/IDEA-Research/HumanSD)] [[Project Page](https://idea-research.github.io/HumanSD/)]
* [CVPR 2023] **Learning Locally Editable Virtual Humans** [[Paper](https://arxiv.org/abs/2305.00121)] [[GitHub](https://github.com/custom-humans/editable-humans)] [[Project Page](https://custom-humans.github.io/)] [[Dataset](https://custom-humans.ait.ethz.ch/)]
* [CVPR 2023] **Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation** [[Paper](https://arxiv.org/abs/2311.17117)] [[GitHub](https://github.com/HumanAIGC/AnimateAnyone)] [[Project Page](https://humanaigc.github.io/animate-anyone/)]
* [CVPRW 2024] **LatentMan: Generating Consistent Animated Characters using Image Diffusion Models** [[Paper](https://arxiv.org/abs/2312.07133)] [[GitHub](https://github.com/abdo-eldesokey/latentman)] [[Project Page](https://abdo-eldesokey.github.io/latentman/)]
* [IJCAI 2024] **Zero-shot High-fidelity and Pose-controllable Character Animation** [[Paper](https://arxiv.org/abs/2404.13680)] 
* [arXiv 2024] **UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation** [[Paper](https://arxiv.org/abs/2406.01188)] [[GitHub](https://github.com/ali-vilab/UniAnimate)] [[Project Page](https://unianimate.github.io/)]
* [arXiv 2024] **MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling** [[Paper](https://arxiv.org/abs/2409.16160)] [[GitHub](https://github.com/menyifang/MIMO)] [[Project Page](https://menyifang.github.io/projects/MIMO/index.html)]

----

### 3D Generation

#### 3D Algorithms
##### Text-to-3D Generation.
>##### Survey
* [arXiv 2023] **Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era** [[Paper](https://arxiv.org/abs/2305.06131)]
* [arXiv 2024] **Advances in 3D Generation: A Survey** [[Paper](https://arxiv.org/abs/2401.17807)]
* [arXiv 2024] **A Survey On Text-to-3D Contents Generation In The Wild** [[Paper](https://arxiv.org/abs/2405.09431)]

>##### Feedforward Approaches.
* [arXiv 2022] **3D-LDM: Neural Implicit 3D Shape Generation with Latent Diffusion Models** [[Paper](https://arxiv.org/abs/2212.00842)] [[GitHub](https://www.3dldm.org/)] 
* [arXiv 2022] **Point-E: A System for Generating 3D Point Clouds from Complex Prompts** [[Paper](https://arxiv.org/abs/2212.08751)] [[GitHub](https://github.com/openai/point-e)] 
* [arXiv 2023] **Shap-E: Generating Conditional 3D Implicit Functions** [[Paper](https://arxiv.org/abs/2305.02463)] [[GitHub](https://github.com/openai/shap-e)] 
* [NeurIPS 2023] **Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation** [[Paper](https://arxiv.org/abs/2306.17115)] [[GitHub](https://github.com/NeuralCarver/Michelangelo)] [[Project Page](https://neuralcarver.github.io/michelangelo/)]
* [ICCV 2023] **ATT3D: Amortized Text-to-3D Object Synthesis** [[Paper](https://arxiv.org/abs/2306.07349)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/ATT3D/)]
* [ICLR 2023 Spotlight] **MeshDiffusion: Score-based Generative 3D Mesh Modeling** [[Paper](https://arxiv.org/abs/2303.08133)] [[GitHub](https://github.com/lzzcd001/MeshDiffusion/)] [[Project Page](https://meshdiffusion.github.io/)]
* [CVPR 2023] **Diffusion-SDF: Text-to-Shape via Voxelized Diffusion** [[Paper](https://arxiv.org/abs/2212.03293)] [[GitHub](https://github.com/ttlmh/Diffusion-SDF)] [[Project Page](https://ttlmh.github.io/DiffusionSDF/)]
* [ICML 2024] **HyperFields:Towards Zero-Shot Generation of NeRFs from Text** [[Paper](https://arxiv.org/abs/2310.17075)] [[GitHub](https://github.com/threedle/hyperfields)] [[Project Page](https://threedle.github.io/hyperfields/)]
* [ECCV 2024] **LATTE3D: Large-scale Amortized Text-To-Enhanced3D Synthesis** [[Paper](https://arxiv.org/abs/2403.15385)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/LATTE3D/)]
* [arXiv 2024] **AToM: Amortized Text-to-Mesh using 2D Diffusion** [[Paper](https://arxiv.org/abs/2402.00867)] [[GitHub](https://github.com/snap-research/AToM)] [[Project Page](https://snap-research.github.io/AToM/)]

>##### Optimization-based Approaches.
* [ICLR 2023 notable top 5%] **DreamFusion: Text-to-3D using 2D Diffusion** [[Paper](https://arxiv.org/abs/2209.14988)] [[Project Page](https://dreamfusion3d.github.io/)]
* [CVPR 2023 Highlight] **Magic3D: High-Resolution Text-to-3D Content Creation** [[Paper](https://arxiv.org/abs/2211.10440)] [[Project Page](https://research.nvidia.com/labs/dir/magic3d/)]
* [CVPR 2023] **Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models** [[Paper](https://arxiv.org/abs/2212.14704)] [[Project Page](https://bluestyle97.github.io/dream3d/)]
* [ICCV 2023] **Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation** [[Paper](https://arxiv.org/abs/2303.13873)] [[GitHub](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)] [[Project Page](https://fantasia3d.github.io/)]
* [NeurIPS 2023 Spotlight] **ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation** [[Paper](https://arxiv.org/abs/2305.16213)] [[GitHub](https://github.com/thu-ml/prolificdreamer)] [[Project Page](https://ml.cs.tsinghua.edu.cn/prolificdreamer/)]
* [ICLR 2024 Poster] **MVDream: Multi-view Diffusion for 3D Generation** [[Paper](https://arxiv.org/abs/2308.16512)] [[GitHub](https://github.com/bytedance/MVDream)] [[Project Page](https://mv-dream.github.io/)]
* [ICLR 2024 Oral] **DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation** [[Paper](https://arxiv.org/abs/2309.16653)] [[GitHub](https://github.com/dreamgaussian/dreamgaussian)] [[Project Page](https://dreamgaussian.github.io/)]
* [CVPR 2024] **PI3D: Efficient Text-to-3D Generation with Pseudo-Image Diffusion** [[Paper](https://arxiv.org/abs/2312.09069)] 
* [CVPR 2024] **VP3D: Unleashing 2D Visual Prompt for Text-to-3D Generation** [[Paper](https://arxiv.org/abs/2403.17001)] [[Project Page](https://vp3d-cvpr24.github.io/)]
* [CVPR 2024] **GSGEN: Text-to-3D using Gaussian Splatting** [[Paper](https://arxiv.org/abs/2309.16585)]  [[GitHub](https://github.com/gsgen3d/gsgen)] [[Project Page](https://gsgen3d.github.io/)]
* [CVPR 2024] **GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models** [[Paper](https://arxiv.org/abs/2310.08529)]  [[GitHub](https://github.com/hustvl/GaussianDreamer)] [[Project Page](https://taoranyi.com/gaussiandreamer/)]
* [CVPR 2024] **Sculpt3D: Multi-View Consistent Text-to-3D Generation with Sparse 3D Prior** [[Paper](https://arxiv.org/abs/2403.09140)]  [[GitHub](https://github.com/StellarCheng/Scuplt_3d/tree/main)] [[Project Page](https://stellarcheng.github.io/Sculpt3D/)]

>##### MVS-based Approaches.
* [ICLR 2024 Poster] **Instant3D: Fast Text-to-3D with Sparse-view Generation and Large Reconstruction Model** [[Paper](https://arxiv.org/abs/2311.06214)] [[Project Page](https://jiahao.ai/instant3d/)]
* [CVPR 2024] **Direct2.5: Diverse Text-to-3D Generation via Multi-view 2.5D Diffusion** [[Paper](https://arxiv.org/abs/2311.15980)]  [[GitHub](https://github.com/apple/ml-direct2.5)] [[Project Page](https://nju-3dv.github.io/projects/direct25/)]
* [CVPR 2024] **Sherpa3D: Boosting High-Fidelity Text-to-3D Generation via Coarse 3D Prior** [[Paper](https://arxiv.org/abs/2312.06655)]  [[GitHub](https://github.com/liuff19/Sherpa3D)] [[Project Page](https://liuff19.github.io/Sherpa3D/)]

##### Image-to-3D Generation.
>##### Feedforward Approaches.
* [arXiv 2023] **3DGen: Triplane Latent Diffusion for Textured Mesh Generation** [[Paper](https://arxiv.org/abs/2303.05371)] 
* [NeurIPS 2023] **Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation** [[Paper](https://arxiv.org/abs/2306.17115)] [[GitHub](https://github.com/NeuralCarver/Michelangelo)] [[Project Page](https://neuralcarver.github.io/michelangelo/)]
* [NeurIPS 2024] **Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer** [[Paper](https://arxiv.org/abs/2405.14832)] [[GitHub](https://github.com/DreamTechAI/Direct3D)] [[Project Page](https://www.neural4d.com/research/direct3d)]
* [SIGGRAPH 2024 Best Paper Honorable Mention] **CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets** [[Paper](https://arxiv.org/abs/2406.13897)] [[GitHub](https://github.com/CLAY-3D/OpenCLAY)] [[Project Page](https://sites.google.com/view/clay-3dlm)]
* [arXiv 2024] **CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner** [[Paper](https://arxiv.org/abs/2405.14979)] [[GitHub](https://github.com/wyysf-98/CraftsMan3D)] [[Project Page](https://craftsman3d.github.io/)]
* [arXiv 2024] **Structured 3D Latents for Scalable and Versatile 3D Generation** [[Paper](https://arxiv.org/abs/2412.01506)] [[GitHub](https://github.com/Microsoft/TRELLIS)] [[Project Page](https://trellis3d.github.io/)]

>##### Optimization-based Approaches.
* [arXiv 2023] **Consistent123: Improve Consistency for One Image to 3D Object Synthesis** [[Paper](https://arxiv.org/abs/2310.08092)] [[Project Page](https://consistent-123.github.io/)]
* [arXiv 2023] **ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation** [[Paper](https://arxiv.org/abs/2312.02201)] [[GitHub](https://github.com/bytedance/ImageDream)] [[Project Page](https://image-dream.github.io/)]
* [CVPR 2023] **RealFusion: 360° Reconstruction of Any Object from a Single Image** [[Paper](https://arxiv.org/abs/2302.10663)] [[GitHub](https://github.com/lukemelas/realfusion)] [[Project Page](https://lukemelas.github.io/realfusion/)]
* [ICCV 2023] **Zero-1-to-3: Zero-shot One Image to 3D Object** [[Paper](https://arxiv.org/abs/2303.11328)] [[GitHub](https://github.com/cvlab-columbia/zero123)] [[Project Page](https://zero123.cs.columbia.edu/)]
* [ICLR 2024 Poster] **Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors** [[Paper](https://arxiv.org/abs/2306.17843)] [[GitHub](https://github.com/guochengqian/Magic123)] [[Project Page](https://guochengqian.github.io/project/magic123/)]
* [ICLR 2024 Poster] **TOSS: High-quality Text-guided Novel View Synthesis from a Single Image** [[Paper](https://arxiv.org/abs/2310.10644)] [[GitHub](https://github.com/IDEA-Research/TOSS)] [[Project Page](https://toss3d.github.io/)]
* [ICLR 2024 Spotlight] **SyncDreamer: Generating Multiview-consistent Images from a Single-view Image** [[Paper](https://arxiv.org/abs/2309.03453)] [[GitHub](https://github.com/liuyuan-pal/SyncDreamer)] [[Project Page](https://liuyuan-pal.github.io/SyncDreamer/)]
* [CVPR 2024] **Wonder3D: Single Image to 3D using Cross-Domain Diffusion** [[Paper](https://arxiv.org/abs/2310.15008)]  [[GitHub](https://github.com/xxlong0/Wonder3D)] [[Project Page](https://www.xxlong.site/Wonder3D/)]

>##### MVS-based Approaches.
* [NeurIPS 2023] **One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization** [[Paper](https://arxiv.org/abs/2306.16928)] [[GitHub](https://github.com/One-2-3-45/One-2-3-45)] [[Project Page](https://one-2-3-45.github.io/)]
* [ECCV 2024] **CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model** [[Paper](https://arxiv.org/abs/2403.05034)] [[GitHub](https://github.com/thu-ml/CRM)] [[Project Page](https://ml.cs.tsinghua.edu.cn/~zhengyi/CRM/)]
* [arXiv 2024] **InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models** [[Paper](https://arxiv.org/abs/2404.07191)] [[GitHub](https://github.com/TencentARC/InstantMesh)]
* [ICLR 2024 Oral] **LRM: Large Reconstruction Model for Single Image to 3D** [[Paper](https://arxiv.org/abs/2311.04400)] [[Project Page](https://yiconghong.me/LRM/)]
* [NeurIPS 2024] **Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image** [[Paper](https://arxiv.org/abs/2405.20343)] [[GitHub](https://github.com/AiuniAI/Unique3D)] [[Project Page](https://wukailu.github.io/Unique3D/)]


##### Video-to-3D Generation.
* [CVPR 2024 Highlight] **ViVid-1-to-3: Novel View Synthesis with Video Diffusion Models** [[Paper](https://arxiv.org/abs/2312.01305)] [[GitHub](https://github.com/ubc-vision/vivid123)] [[Project Page](https://jgkwak95.github.io/ViVid-1-to-3/)]
* [ICML 2024] **IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation** [[Paper](https://arxiv.org/abs/2402.08682)] [[Project Page](https://lukemelas.github.io/IM-3D/)]
* [arXiv 2024] **V3D: Video Diffusion Models are Effective 3D Generators** [[Paper](https://arxiv.org/abs/2403.06738)] [[GitHub](https://github.com/heheyas/V3D)] [[Project Page](https://heheyas.github.io/V3D/)]
* [ECCV 2024 Oral] **SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image Using Latent Video Diffusion** [[Paper](https://arxiv.org/abs/2403.12008)] [[Project Page](https://sv3d.github.io/)]
* [NeurIPS 2024 Oral] **CAT3D: Create Anything in 3D with Multi-View Diffusion Models** [[Paper](https://arxiv.org/abs/2405.10314)] [[Project Page](https://cat3d.github.io/)]

#### 3D Applications
>##### Avatar Generation.
* [CVPR 2023] **Zero-Shot Text-to-Parameter Translation for Game Character Auto-Creation** [[Paper](https://arxiv.org/abs/2303.01311)]
* [SIGGRAPH 2023] **DreamFace: Progressive Generation of Animatable 3D Faces under Text Guidance** [[Paper](https://arxiv.org/abs/2304.03117)] [[Project Page](https://sites.google.com/view/dreamface)]
* [NeurIPS 2023] **Headsculpt: Crafting 3d head avatars with text** [[Paper](https://arxiv.org/abs/2306.03038)] [[GitHub](https://github.com/BrandonHanx/HeadSculpt)] [[Project Page](https://brandonhan.uk/HeadSculpt/)]
* [NeurIPS 2023] **DreamWaltz: Make a Scene with Complex 3D Animatable Avatars** [[Paper](https://arxiv.org/abs/2305.12529)] [[GitHub](https://github.com/IDEA-Research/DreamWaltz)] [[Project Page](https://idea-research.github.io/DreamWaltz/)]
* [NeurIPS 2023 Spotlight] **DreamHuman: Animatable 3D Avatars from Text** [[Paper](https://arxiv.org/abs/2306.09329)] [[Project Page](https://dream-human.github.io/)]

>##### Scene Generation. 
* [ACM MM 2023] **RoomDreamer: Text-Driven 3D Indoor Scene Synthesis with Coherent Geometry and Texture** [[Paper](https://arxiv.org/abs/2305.11337)]
* [TVCG 2024] **Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields** [[Paper](https://arxiv.org/abs/2305.11588)] [[GitHub](https://github.com/eckertzhang/Text2NeRF)] [[Project Page](https://eckertzhang.github.io/Text2NeRF.github.io/)]
* [ECCV 2024] **DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling** [[Paper](https://arxiv.org/abs/2404.03575)] [[GitHub](https://github.com/DreamScene-Project/DreamScene)] [[Project Page](https://dreamscene-project.github.io/)]
* [ECCV 2024] **DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting** [[Paper](https://arxiv.org/abs/2404.06903)] [[GitHub](https://github.com/ShijieZhou-UCLA/DreamScene360)] [[Project Page](https://dreamscene360.github.io/)]
* [arXiv 2024] **Urban Architect: Steerable 3D Urban Scene Generation with Layout Prior** [[Paper](https://arxiv.org/abs/2404.06780)] [[GitHub](https://github.com/UrbanArchitect/UrbanArchitect)] [[Project Page](https://urbanarchitect.github.io/)]
* [arXiv 2024] **CityCraft: A Real Crafter for 3D City Generation** [[Paper](https://arxiv.org/abs/2406.04983)] [[GitHub](https://github.com/djFatNerd/CityCraft)]

>##### 3D Editing. 
* [ECCV 2022] **Unified Implicit Neural Stylization** [[Paper](https://arxiv.org/abs/2204.01943)] [[GitHub](https://github.com/VITA-Group/INS)] [[Project Page](https://zhiwenfan.github.io/INS/)]
* [ECCV 2022] **ARF: Artistic Radiance Fields** [[Paper](https://arxiv.org/abs/2206.06360)] [[GitHub](https://github.com/Kai-46/ARF-svox2)] [[Project Page](https://www.cs.cornell.edu/projects/arf/)]
* [SIGGRAPH Asia 2022] **FDNeRF: Few-shot Dynamic Neural Radiance Fields for Face Reconstruction and Expression Editing** [[Paper](https://arxiv.org/abs/2208.05751)] [[GitHub](https://github.com/FDNeRF/FDNeRF)] [[Project Page](https://fdnerf.github.io/)]
* [CVPR 2022] **FENeRF: Face Editing in Neural Radiance Fields** [[Paper](https://arxiv.org/abs/2111.15490)] [[GitHub](https://github.com/MrTornado24/FENeRF)] [[Project Page](https://mrtornado24.github.io/FENeRF/)]
* [SIGGRAPH 2023] **TextDeformer: Geometry Manipulation using Text Guidance** [[Paper](https://arxiv.org/abs/2304.13348)] [[GitHub](https://github.com/threedle/TextDeformer)] [[Project Page](https://threedle.github.io/TextDeformer/)]
* [ICCV 2023] **ObjectSDF++: Improved Object-Compositional Neural Implicit Surfaces** [[Paper](https://arxiv.org/abs/2308.07868)] [[GitHub](https://github.com/QianyiWu/objectsdf_plus)] [[Project Page](https://wuqianyi.top/objectsdf++)] 
* [ICCV 2023 Oral] **Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions** [[Paper](https://arxiv.org/abs/2303.12789)] [[GitHub](https://github.com/ayaanzhaque/instruct-nerf2nerf)] [[Project Page](https://instruct-nerf2nerf.github.io/)] 

----

### 4D Generation

#### 4D Algorithms
>##### Feedforward Approaches.
* [CVPR 2024] **Control4D: Efficient 4D Portrait Editing with Text** [[Paper](https://arxiv.org/abs/2305.20082)] [[Project Page](https://control4darxiv.github.io/)]
* [NeurIPS 2024] **Animate3D: Animating Any 3D Model with Multi-view Video Diffusion** [[Paper](https://arxiv.org/abs/2407.11398)] [[GitHub](https://github.com/yanqinJiang/Animate3D)] [[Project Page](https://animate3d.github.io/)]
* [NeurIPS 2024] **Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels** [[Paper](https://arxiv.org/abs/2405.16822)] [[GitHub](https://github.com/yikaiw/vidu4d)] [[Project Page](https://vidu4d-dgs.github.io/)]
* [NeurIPS 2024] **Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models** [[Paper](https://arxiv.org/abs/2405.16645)] [[GitHub](https://github.com/VITA-Group/Diffusion4D)] [[Project Page](https://vita-group.github.io/Diffusion4D/)] [[Dataset](https://huggingface.co/datasets/hw-liang/Diffusion4D)]
* [NeurIPS 2024] **L4GM: Large 4D Gaussian Reconstruction Model** [[Paper](https://arxiv.org/abs/2406.10324)] [[GitHub](https://github.com/nv-tlabs/L4GM-official)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/l4gm/)] 

>##### Optimization-based Approaches.
* [arXiv 2023] **Text-To-4D Dynamic Scene Generation** [[Paper](https://arxiv.org/abs/2301.11280)] [[Project Page](https://make-a-video3d.github.io/)]
* [CVPR 2024] **4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling** [[Paper](https://arxiv.org/abs/2311.17984)] [[GitHub](https://github.com/sherwinbahmani/4dfy)] [[Project Page](https://sherwinbahmani.github.io/4dfy/)]
* [CVPR 2024] **A Unified Approach for Text- and Image-guided 4D Scene Generation** [[Paper](https://arxiv.org/abs/2311.16854)] [[GitHub](https://github.com/NVlabs/dream-in-4d)] [[Project Page](https://research.nvidia.com/labs/nxp/dream-in-4d/)]
* [CVPR 2024] **Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models** [[Paper](https://arxiv.org/abs/2312.13763)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/AlignYourGaussians/)]
* [ECCV 2024] **TC4D: Trajectory-Conditioned Text-to-4D Generation** [[Paper](https://arxiv.org/abs/2403.17920)] [[GitHub](https://github.com/sherwinbahmani/tc4d)] [[Project Page](https://sherwinbahmani.github.io/tc4d/)]
* [ECCV 2024] **SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer** [[Paper](https://arxiv.org/abs/2404.03736)] [[GitHub](https://github.com/JarrentWu1031/SC4D)] [[Project Page](https://sc4d.github.io/)]
* [ECCV 2024] **STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians** [[Paper](https://arxiv.org/abs/2403.14939)] [[GitHub](https://github.com/zeng-yifei/STAG4D)] [[Project Page](https://nju-3dv.github.io/projects/STAG4D/)]
* [NeurIPS 2024] **4Real: Towards Photorealistic 4D Scene Generation via Video Diffusion Models** [[Paper](https://arxiv.org/abs/2406.07472)] [[Project Page](https://snap-research.github.io/4Real/)]
* [NeurIPS 2024] **Compositional 3D-aware Video Generation with LLM Director** [[Paper](https://arxiv.org/abs/2409.00558)] [[Project Page](https://www.microsoft.com/en-us/research/project/compositional-3d-aware-video-generation/)]
* [NeurIPS 2024] **DreamScene4D: Dynamic Multi-Object Scene Generation from Monocular Videos** [[Paper](https://arxiv.org/abs/2405.02280)] [[GitHub](https://github.com/dreamscene4d/dreamscene4d)] [[Project Page](https://dreamscene4d.github.io/)]
* [NeurIPS 2024] **DreamMesh4D: Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation** [[Paper](https://arxiv.org/abs/2410.06756)] [[GitHub](https://github.com/WU-CVGL/DreamMesh4D)] [[Project Page](https://lizhiqi49.github.io/DreamMesh4D/)]

#### 4D Applications
>##### 4D Editing. 
* [CVPR 2024] **Control4D: Efficient 4D Portrait Editing with Text** [[Paper](https://arxiv.org/abs/2305.20082)] [[Project Page](https://control4darxiv.github.io/)]
* [CVPR 2024] **Instruct 4D-to-4D: Editing 4D Scenes as Pseudo-3D Scenes Using 2D Diffusion** [[Paper](https://arxiv.org/abs/2406.09402)] [[GitHub](https://github.com/Friedrich-M/Instruct-4D-to-4D/)] [[Project Page](https://immortalco.github.io/Instruct-4D-to-4D/)]

>##### Human Animation.
* [SIGGRAPH 2020] **Robust Motion In-betweening** [[Paper](https://arxiv.org/abs/2102.04942)]
* [CVPR 2022] **Generating Diverse and Natural 3D Human Motions from Text** [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf)] [[GitHub](https://github.com/EricGuo5513/text-to-motion)] [[Project Page](https://ericguo5513.github.io/text-to-motion/)]
* [SCA 2023] **Motion In-Betweening with Phase Manifolds** [[Paper](https://arxiv.org/abs/2308.12751)] [[GitHub](https://github.com/paulstarke/PhaseBetweener)]
* [CVPR 2023] **T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations** [[Paper](https://arxiv.org/abs/2301.06052)] [[GitHub](https://github.com/Mael-zys/T2M-GPT)] [[Project Page](https://mael-zys.github.io/T2M-GPT/)]
* [ICLR 2023 notable top 25%] **Human Motion Diffusion Model** [[Paper](https://arxiv.org/abs/2209.14916)] [[GitHub](https://github.com/GuyTevet/motion-diffusion-model)] [[Project Page](https://guytevet.github.io/mdm-page/)]
* [NeurIPS 2023] **MotionGPT: Human Motion as a Foreign Language** [[Paper](https://arxiv.org/abs/2306.14795)] [[GitHub](https://github.com/OpenMotionLab/MotionGPT)] [[Project Page](https://motion-gpt.github.io/)]
* [ICML 2024] **HumanTOMATO: Text-aligned Whole-body Motion Generation** [[Paper](https://arxiv.org/abs/2310.12978)] [[GitHub](https://github.com/IDEA-Research/HumanTOMATO)] [[Project Page](https://lhchen.top/HumanTOMATO/)]
* [CVPR 2024] **MoMask: Generative Masked Modeling of 3D Human Motions** [[Paper](https://arxiv.org/abs/2312.00063)] [[GitHub](https://github.com/EricGuo5513/momask-codes)] [[Project Page](https://ericguo5513.github.io/momask/)]
* [CVPR 2024] **Lodge: A Coarse to Fine Diffusion Network for Long Dance Generation Guided by the Characteristic Dance Primitives** [[Paper](https://arxiv.org/abs/2403.10518)] [[GitHub](https://github.com/li-ronghui/LODGE)] [[Project Page](https://li-ronghui.github.io/lodge)]

-------

## 🔥 Awesome Text2X Resources

An open collection of state-of-the-art (SOTA), novel **Text to X (X can be everything)** methods (papers, codes and datasets), intended to keep pace with the anticipated surge of research.

<div><div align="center">
	<img width="500" height="350" src="media/logo.svg" alt="Awesome"></div>



## Update Logs
* `2025.02.28` - update several papers status "CVPR 2025" to accepted papers, congrats to all 🎉
  
<details span>
<summary><b>2025 Update Logs:</b></summary>
<br>	
* `2025.01.23` - update several papers status "ICLR 2025" to accepted papers, congrats to all 🎉
* `2025.01.09` - update layout.

  
</details>

<details close>
<summary><b>Previous 2024 Update Logs:</b></summary>
* `2024.12.21` adjusted the layouts of several sections and _Happy Winter Solstice_ ⚪🥣.
* `2024.09.26` - update several papers status "NeurIPS 2024" to accepted papers, congrats to all 🎉
* `2024.09.03` - add one new section 'text to model'.
* `2024.06.30` - add one new section 'text to video'.	
* `2024.07.02` - update several papers status "ECCV 2024" to accepted papers, congrats to all 🎉
* `2024.06.21` - add one hot Topic about _AIGC 4D Generation_ on the section of __Suvery and Awesome Repos__.
* `2024.06.17` - an awesome repo for CVPR2024 [Link](https://github.com/52CV/CVPR-2024-Papers) 👍🏻
* `2024.04.05` adjusted the layout and added accepted lists and ArXiv lists to each section.
* `2024.04.05` - an awesome repo for CVPR2024 on 3DGS and NeRF [Link](https://github.com/Yubel426/NeRF-3DGS-at-CVPR-2024) 👍🏻
* `2024.03.25` - add one new survey paper of 3D GS into the section of "Survey and Awesome Repos--Topic 1: 3D Gaussian Splatting".
* `2024.03.12` - add a new section "Dynamic Gaussian Splatting", including Neural Deformable 3D Gaussians, 4D Gaussians, Dynamic 3D Gaussians.
* `2024.03.11` - CVPR 2024 Accpeted Papers [Link](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers) 
* update some papers accepted by CVPR 2024! Congratulations🎉
  
</details>
<br>

## Text to 4D
(Also, Image/Video to 4D)

### 💡 4D ArXiv Papers

#### 1. AR4D: Autoregressive 4D Generation from Monocular Videos
Hanxin Zhu, Tianyu He, Xiqian Yu, Junliang Guo, Zhibo Chen, Jiang Bian (University of Science and Technology of China, Microsoft Research Asia)
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in generative models have ignited substantial interest in dynamic 3D content creation (\ie, 4D generation). Existing approaches primarily rely on Score Distillation Sampling (SDS) to infer novel-view videos, typically leading to issues such as limited diversity, spatial-temporal inconsistency and poor prompt alignment, due to the inherent randomness of SDS. To tackle these problems, we propose AR4D, a novel paradigm for SDS-free 4D generation. Specifically, our paradigm consists of three stages. To begin with, for a monocular video that is either generated or captured, we first utilize pre-trained expert models to create a 3D representation of the first frame, which is further fine-tuned to serve as the canonical space. Subsequently, motivated by the fact that videos happen naturally in an autoregressive manner, we propose to generate each frame's 3D representation based on its previous frame's representation, as this autoregressive generation manner can facilitate more accurate geometry and motion estimation. Meanwhile, to prevent overfitting during this process, we introduce a progressive view sampling strategy, utilizing priors from pre-trained large-scale 3D reconstruction models. To avoid appearance drift introduced by autoregressive generation, we further incorporate a refinement stage based on a global deformation field and the geometry of each frame's 3D representation. Extensive experiments have demonstrated that AR4D can achieve state-of-the-art 4D generation without SDS, delivering greater diversity, improved spatial-temporal consistency, and better alignment with input prompts.
</details>

#### 2. GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking
Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yijin Li, Fu-Yun Wang, Hongsheng Li

(The Chinese University of Hong Kong, Centre for Perceptual and Interactive Intelligence, Avolution AI)
<details span>
<summary><b>Abstract</b></summary>
4D video control is essential in video generation as it enables the use of sophisticated lens techniques, such as multi-camera shooting and dolly zoom, which are currently unsupported by existing methods. Training a video Diffusion Transformer (DiT) directly to control 4D content requires expensive multi-view videos. Inspired by Monocular Dynamic novel View Synthesis (MDVS) that optimizes a 4D representation and renders videos according to different 4D elements, such as camera pose and object motion editing, we bring pseudo 4D Gaussian fields to video generation. Specifically, we propose a novel framework that constructs a pseudo 4D Gaussian field with dense 3D point tracking and renders the Gaussian field for all video frames. Then we finetune a pretrained DiT to generate videos following the guidance of the rendered video, dubbed as GS-DiT. To boost the training of the GS-DiT, we also propose an efficient Dense 3D Point Tracking (D3D-PT) method for the pseudo 4D Gaussian field construction. Our D3D-PT outperforms SpatialTracker, the state-of-the-art sparse 3D point tracking method, in accuracy and accelerates the inference speed by two orders of magnitude. During the inference stage, GS-DiT can generate videos with the same dynamic content while adhering to different camera parameters, addressing a significant limitation of current video generation models. GS-DiT demonstrates strong generalization capabilities and extends the 4D controllability of Gaussian splatting to video generation beyond just camera poses. It supports advanced cinematic effects through the manipulation of the Gaussian field and camera intrinsics, making it a powerful tool for creative video production.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **AR4D: Autoregressive 4D Generation from Monocular Videos**  | 3 Jan 2025 |          [Link](https://arxiv.org/abs/2501.01722)          | --  | [Link](https://hanxinzhu-lab.github.io/AR4D/)  |
| 2025 | **GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking**  | 5 Jan 2025 |          [Link](https://arxiv.org/abs/2501.02690)          | [Link](https://github.com/wkbian/GS-DiT)  | [Link](https://wkbian.github.io/Projects/GS-DiT/)  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@misc{zhu2025ar4dautoregressive4dgeneration,
      title={AR4D: Autoregressive 4D Generation from Monocular Videos}, 
      author={Hanxin Zhu and Tianyu He and Xiqian Yu and Junliang Guo and Zhibo Chen and Jiang Bian},
      year={2025},
      eprint={2501.01722},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.01722}, 
}

@article{bian2025gsdit,
  title={GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking},
  author={Bian, Weikang and Huang, Zhaoyang and Shi, Xiaoyu and and Li, Yijin and Wang, Fu-Yun and Li, Hongsheng},
  journal={arXiv preprint arXiv:2501.02690},
  year={2025}
}

```
</details>

---

### Previous Papers

### Year 2023
In 2023, tasks classified as text/Image to 4D and video to 4D generally involve producing four-dimensional data from text/Image or video input. For more details, please check the [2023 4D Papers](./docs/4d/4d_2023.md), including 6 accepted papers and 3 arXiv papers.

### Year 2024
For more details, please check the [2024 4D Papers](./docs/4d/4d_2024.md), including 17 accepted papers and 17 arXiv papers.

--------------


## Text to Video

### 💡 T2V ArXiv Papers

#### 1. TransPixar: Advancing Text-to-Video Generation with Transparency
Luozhou Wang, Yijun Li, Zhifei Chen, Jui-Hsien Wang, Zhifei Zhang, He Zhang, Zhe Lin, Yingcong Chen

(HKUST(GZ), HKUST, Adobe Research)
<details span>
<summary><b>Abstract</b></summary>
Text-to-video generative models have made significant strides, enabling diverse applications in entertainment, advertising, and education. However, generating RGBA video, which includes alpha channels for transparency, remains a challenge due to limited datasets and the difficulty of adapting existing models. Alpha channels are crucial for visual effects (VFX), allowing transparent elements like smoke and reflections to blend seamlessly into scenes. We introduce TransPixar, a method to extend pretrained video models for RGBA generation while retaining the original RGB capabilities. TransPixar leverages a diffusion transformer (DiT) architecture, incorporating alpha-specific tokens and using LoRA-based fine-tuning to jointly generate RGB and alpha channels with high consistency. By optimizing attention mechanisms, TransPixar preserves the strengths of the original RGB model and achieves strong alignment between RGB and alpha channels despite limited training data. Our approach effectively generates diverse and consistent RGBA videos, advancing the possibilities for VFX and interactive content creation.
</details>

#### 2. Multi-subject Open-set Personalization in Video Generation
Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, Sergey Tulyakov

(Snap Inc., UC Merced, CMU)
<details span>
<summary><b>Abstract</b></summary>
Video personalization methods allow us to synthesize videos with specific concepts such as people, pets, and places. However, existing methods often focus on limited domains, require time-consuming optimization per subject, or support only a single subject. We present Video Alchemist − a video model with built-in multi-subject, open-set personalization capabilities for both foreground objects and background, eliminating the need for time-consuming test-time optimization. Our model is built on a new Diffusion Transformer module that fuses each conditional reference image and its corresponding subject-level text prompt with cross-attention layers. Developing such a large model presents two main challenges: dataset and evaluation. First, as paired datasets of reference images and videos are extremely hard to collect, we sample selected video frames as reference images and synthesize a clip of the target video. However, while models can easily denoise training videos given reference frames, they fail to generalize to new contexts. To mitigate this issue, we design a new automatic data construction pipeline with extensive image augmentations. Second, evaluating open-set video personalization is a challenge in itself. To address this, we introduce a personalization benchmark that focuses on accurate subject fidelity and supports diverse personalization scenarios. Finally, our extensive experiments show that our method significantly outperforms existing personalization methods in both quantitative and qualitative evaluations.
</details>

#### 3. BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations
Weixi Feng, Chao Liu, Sifei Liu, William Yang Wang, Arash Vahdat, Weili Nie (UC Santa Barbara, NVIDIA)
<details span>
<summary><b>Abstract</b></summary>
Existing video generation models struggle to follow complex text prompts and synthesize multiple objects, raising the need for additional grounding input for improved controllability. In this work, we propose to decompose videos into visual primitives - blob video representation, a general representation for controllable video generation. Based on blob conditions, we develop a blob-grounded video diffusion model named BlobGEN-Vid that allows users to control object motions and fine-grained object appearance. In particular, we introduce a masked 3D attention module that effectively improves regional consistency across frames. In addition, we introduce a learnable module to interpolate text embeddings so that users can control semantics in specific frames and obtain smooth object transitions. We show that our framework is model-agnostic and build BlobGEN-Vid based on both U-Net and DiT-based video diffusion models. Extensive experimental results show that BlobGEN-Vid achieves superior zero-shot video generation ability and state-of-the-art layout controllability on multiple benchmarks. When combined with an LLM for layout planning, our framework even outperforms proprietary text-to-video generators in terms of compositional accuracy.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **TransPixar: Advancing Text-to-Video Generation with Transparency**  | 6 Jan 2025 |          [Link](https://arxiv.org/abs/2501.03006)          | [Link](https://github.com/wileewang/TransPixar)  | [Link](https://wileewang.github.io/TransPixar/)  |
| 2025 | **Multi-subject Open-set Personalization in Video Generation**  | 10 Jan 2025 |          [Link](https://arxiv.org/abs/2501.06187)          | -- | [Link](https://snap-research.github.io/open-set-video-personalization/)  |
| 2025 | **BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations**  | 13 Jan 2025 |          [Link](https://arxiv.org/abs/2501.07647)          | -- | [Link](https://blobgen-vid2.github.io/)  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@misc{wang2025transpixar,
     title={TransPixar: Advancing Text-to-Video Generation with Transparency}, 
     author={Luozhou Wang and Yijun Li and Zhifei Chen and Jui-Hsien Wang and Zhifei Zhang and He Zhang and Zhe Lin and Yingcong Chen},
     year={2025},
     eprint={2501.03006},
     archivePrefix={arXiv},
     primaryClass={cs.CV},
     url={https://arxiv.org/abs/2501.03006}, 
}

@misc{chen2025multisubjectopensetpersonalizationvideo,
      title={Multi-subject Open-set Personalization in Video Generation}, 
      author={Tsai-Shien Chen and Aliaksandr Siarohin and Willi Menapace and Yuwei Fang and Kwot Sin Lee and Ivan Skorokhodov and Kfir Aberman and Jun-Yan Zhu and Ming-Hsuan Yang and Sergey Tulyakov},
      year={2025},
      eprint={2501.06187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.06187}, 
}

@article{feng2025blobgen,
  title={BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations},
  author={Feng, Weixi and Liu, Chao and Liu, Sifei and Wang, William Yang and Vahdat, Arash and Nie, Weili},
  journal={arXiv preprint arXiv:2501.07647},
  year={2025}
}
```
</details>


---

### Video Other Additional Info

### Previous Papers

### Year 2024
For more details, please check the [2024 T2V Papers](./docs/video/t2v_2024.md), including 16 accepted papers and 11 arXiv papers.

- OSS video generation models: [Mochi 1](https://github.com/genmoai/models) preview is an open state-of-the-art video generation model with high-fidelity motion and strong prompt adherence.
- Survey: The Dawn of Video Generation: Preliminary Explorations with SORA-like Models, [arXiv](https://arxiv.org/abs/2410.05227), [Project Page](https://ailab-cvc.github.io/VideoGen-Eval/), [GitHub Repo](https://github.com/AILab-CVC/VideoGen-Eval)

### 📚 Dataset Works

#### 1. VidGen-1M: A Large-Scale Dataset for Text-to-video Generation
Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, Hao Li

(Fudan University, ShangHai Academy of AI for Science)
<details span>
<summary><b>Abstract</b></summary>
The quality of video-text pairs fundamentally determines the upper bound of text-to-video models. Currently, the datasets used for training these models suffer from significant shortcomings, including low temporal consistency, poor-quality captions, substandard video quality, and imbalanced data distribution. The prevailing video curation process, which depends on image models for tagging and manual rule-based curation, leads to a high computational load and leaves behind unclean data. As a result, there is a lack of appropriate training datasets for text-to-video models. To address this problem, we present VidGen-1M, a superior training dataset for text-to-video models. Produced through a coarse-to-fine curation strategy, this dataset guarantees high-quality videos and detailed captions with excellent temporal consistency. When used to train the video generation model, this dataset has led to experimental results that surpass those obtained with other models.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2024 | **VidGen-1M: A Large-Scale Dataset for Text-to-video Generation**  | 5 Aug 2024  |          [Link](https://arxiv.org/abs/2408.02629)          | [Link](https://github.com/SAIS-FUXI/VidGen) | [Link](https://sais-fuxi.github.io/projects/vidgen-1m/)  |

<details close>
<summary>References</summary>

```
%axiv papers

@article{tan2024vidgen,
  title={VidGen-1M: A Large-Scale Dataset for Text-to-video Generation},
  author={Tan, Zhiyu and Yang, Xiaomeng, and Qin, Luozheng and Li Hao},
  booktitle={arXiv preprint arxiv:2408.02629},
  year={2024}
}


```
</details>

--------------

## Text to Scene

### 💡 3D Scene ArXiv Papers

#### 1. LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation
Yang Zhou, Zongjin He, Qixuan Li, Chao Wang (ShangHai University)
<details span>
<summary><b>Abstract</b></summary>
Recently, the field of text-guided 3D scene generation has garnered significant attention. High-quality generation that aligns with physical realism and high controllability is crucial for practical 3D scene applications. However, existing methods face fundamental limitations: (i) difficulty capturing complex relationships between multiple objects described in the text, (ii) inability to generate physically plausible scene layouts, and (iii) lack of controllability and extensibility in compositional scenes. In this paper, we introduce LayoutDreamer, a framework that leverages 3D Gaussian Splatting (3DGS) to facilitate high-quality, physically consistent compositional scene generation guided by text. Specifically, given a text prompt, we convert it into a directed scene graph and adaptively adjust the density and layout of the initial compositional 3D Gaussians. Subsequently, dynamic camera adjustments are made based on the training focal point to ensure entity-level generation quality. Finally, by extracting directed dependencies from the scene graph, we tailor physical and layout energy to ensure both realism and flexibility. Comprehensive experiments demonstrate that LayoutDreamer outperforms other compositional scene generation quality and semantic alignment methods. Specifically, it achieves state-of-the-art (SOTA) performance in the multiple objects generation metric of T3Bench.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation**  | 4 Feb 2025 |          [Link](https://arxiv.org/abs/2502.01949)          | --  | --  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{zhou2025layoutdreamer,
  title={LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation},
  author={Zhou, Yang and He, Zongjin and Li, Qixuan and Wang, Chao},
  journal={arXiv preprint arXiv:2502.01949},
  year={2025}
}

```
</details>

### Previous Papers

### Year 2023-2024
For more details, please check the [2023-2024 3D Scene Papers](./docs/3d_scene/3d_scene_23-24.md), including 19 accepted papers and 11 arXiv papers.

--------------


## Text to Human Motion

### 💡 Motion ArXiv Papers

#### 1. MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm
Ziyan Guo, Zeyu Hu, Na Zhao, De Wen Soh 

(Singapore University of Technology and Design, LightSpeed Studios)
<details span>
<summary><b>Abstract</b></summary>
Human motion generation and editing are key components of computer graphics and vision. However, current approaches in this field tend to offer isolated solutions tailored to specific tasks, which can be inefficient and impractical for real-world applications. While some efforts have aimed to unify motion-related tasks, these methods simply use different modalities as conditions to guide motion generation. Consequently, they lack editing capabilities, fine-grained control, and fail to facilitate knowledge sharing across tasks. To address these limitations and provide a versatile, unified framework capable of handling both human motion generation and editing, we introduce a novel paradigm: Motion-Condition-Motion, which enables the unified formulation of diverse tasks with three concepts: source motion, condition, and target motion. Based on this paradigm, we propose a unified framework, MotionLab, which incorporates rectified flows to learn the mapping from source motion to target motion, guided by the specified conditions. In MotionLab, we introduce the 1) MotionFlow Transformer to enhance conditional generation and editing without task-specific modules; 2) Aligned Rotational Position Encoding} to guarantee the time synchronization between source motion and target motion; 3) Task Specified Instruction Modulation; and 4) Motion Curriculum Learning for effective multi-task learning and knowledge sharing across tasks. Notably, our MotionLab demonstrates promising generalization capabilities and inference efficiency across multiple benchmarks for human motion.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm**  | 6 Feb 2025 |          [Link](https://arxiv.org/abs/2502.02358)          | [Link](https://github.com/Diouo/MotionLab)  | [Link](https://diouo.github.io/motionlab.github.io/)  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{guo2025motionlab,
  title={MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm},
  author={Guo, Ziyan and Hu, Zeyu and Zhao, Na and Soh, De Wen},
  journal={arXiv preprint arXiv:2502.02358},
  year={2025}
}

```
</details>


---

### Motion Other Additional Info

### Previous Papers

### Year 2023-2024
For more details, please check the [2023-2024 Text to Human Motion Papers](./docs/human_motion/motion_23-24.md), including 31 accepted papers and 13 arXiv papers.

### 📚 Dataset Works

#### Datasets
   | Motion | Info |                              URL                              |               Others                            | 
   | :-----: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
   |  AIST |  AIST Dance Motion Dataset  | [Link](https://aistdancedb.ongaaccel.jp/) |--|
   |  AIST++  |  AIST++ Dance Motion Dataset | [Link](https://google.github.io/aistplusplus_dataset/) | [dance video database with SMPL annotations](https://google.github.io/aistplusplus_dataset/download.html) |
   |  AMASS  |  optical marker-based motion capture datasets  | [Link](https://amass.is.tue.mpg.de/) |--|

#### Additional Info
<details>
<summary>AMASS</summary>

AMASS is a large database of human motion unifying different optical marker-based motion capture datasets by representing them within a common framework and parameterization. AMASS is readily useful for animation, visualization, and generating training data for deep learning.
  
</details>


#### Survey
- Survey: [Human Motion Generation: A Survey](https://arxiv.org/abs/2307.10894), ArXiv 2023 Nov


--------------


## Text to 3D Human

### 💡 Human ArXiv Papers

#### 1. Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars
Tobias Kirschstein, Javier Romero, Artem Sevastopolsky, Matthias Nießner, Shunsuke Saito

(Technical University of Munich, Meta Reality Labs)
<details span>
<summary><b>Abstract</b></summary>
Traditionally, creating photo-realistic 3D head avatars requires a studio-level multi-view capture setup and expensive optimization during test-time, limiting the use of digital human doubles to the VFX industry or offline renderings.
To address this shortcoming, we present Avat3r, which regresses a high-quality and animatable 3D head avatar from just a few input images, vastly reducing compute requirements during inference. More specifically, we make Large Reconstruction Models animatable and learn a powerful prior over 3D human heads from a large multi-view video dataset. For better 3D head reconstructions, we employ position maps from DUSt3R and generalized feature maps from the human foundation model Sapiens. To animate the 3D head, our key discovery is that simple cross-attention to an expression code is already sufficient. Finally, we increase robustness by feeding input images with different expressions to our model during training, enabling the reconstruction of 3D head avatars from inconsistent inputs, e.g., an imperfect phone capture with accidental movement, or frames from a monocular video.
We compare Avat3r with current state-of-the-art methods for few-input and single-input scenarios, and find that our method has a competitive advantage in both tasks. Finally, we demonstrate the wide applicability of our proposed model, creating 3D head avatars from images of different sources, smartphone captures, single images, and even out-of-domain inputs like antique busts.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars**  | 27 Feb 2025 |          [Link](https://arxiv.org/abs/2502.20220)          | --  | [Link](https://tobias-kirschstein.github.io/avat3r/)  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@misc{kirschstein2025avat3r,
      title={Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars},
      author={Tobias Kirschstein and Javier Romero and Artem Sevastopolsky and Matthias Nie\ss{}ner and Shunsuke Saito},
      year={2025},
      eprint={2502.20220},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20220},
}

```
</details>

### Additional Info
### Previous Papers

### Year 2023-2024
For more details, please check the [2023-2024 3D Human Papers](./docs/3d_human/human_23-24.md), including 16 accepted papers and 6 arXiv papers.

<details close>
<summary>Survey and Awesome Repos</summary>
 
#### Survey
- [PROGRESS AND PROSPECTS IN 3D GENERATIVE AI: A TECHNICAL OVERVIEW INCLUDING 3D HUMAN](https://arxiv.org/pdf/2401.02620.pdf), ArXiv 2024
  
#### Awesome Repos
- Resource1: [Awesome Digital Human](https://github.com/weihaox/awesome-digital-human)
</details>

<details close>
<summary>Pretrained Models</summary>

   | Pretrained Models (human body) | Info |                              URL                              |
   | :-----: | :-----: | :----------------------------------------------------------: |
   |  SMPL  |  smpl model (smpl weights) | [Link](https://smpl.is.tue.mpg.de/) |
   |  SMPL-X  |  smpl model (smpl weights)  | [Link](https://smpl-x.is.tue.mpg.de/) |
   |  human_body_prior  |  vposer model (smpl weights)  | [Link](https://github.com/nghorbani/human_body_prior) |
<details>
<summary>SMPL</summary>

SMPL is an easy-to-use, realistic, model of the of the human body that is useful for animation and computer vision.

- version 1.0.0 for Python 2.7 (female/male, 10 shape PCs)
- version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)
- UV map in OBJ format
  
</details>

<details>
<summary>SMPL-X</summary>

SMPL-X, that extends SMPL with fully articulated hands and facial expressions (55 joints, 10475 vertices)

</details>
</details>


--------------

## Related Resources

### Text to 'other tasks'
(Here other tasks refer to *CAD*, *Model* and *Music* etc.)

#### Text to CAD
+ 2024 | CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM | arXiv 7 Nov 2024 | [Paper](https://arxiv.org/abs/2411.04954)  | [Code](https://github.com/CAD-MLLM/CAD-MLLM) | [Project Page](https://cad-mllm.github.io/) 
+ 2024 | Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts | NeurIPS 2024 Spotlight | [Paper](https://arxiv.org/abs/2409.17106)  | [Project Page](https://sadilkhan.github.io/text2cad-project/)

#### Text to Music
+ 2024 | FLUX that Plays Music | arXiv 1 Sep 2024 | [Paper](https://arxiv.org/abs/2409.00587) | [Code](https://github.com/feizc/FluxMusic) | [Hugging Face](https://huggingface.co/feizhengcong/FluxMusic)
</details>

#### Text to Model
+ 2024 | Text-to-Model: Text-Conditioned Neural Network Diffusion for Train-Once-for-All Personalization | arXiv 23 May 2024 | [Paper](https://arxiv.org/abs/2405.14132)



### Survey and Awesome Repos 
<details close>
<summary>🔥 Topic 1: 3D Gaussian Splatting</summary>
 
#### Survey
- [Gaussian Splatting: 3D Reconstruction and Novel View Synthesis, a Review](https://arxiv.org/abs/2405.03417), ArXiv Mon, 6 May 2024
- [Recent Advances in 3D Gaussian Splatting](https://arxiv.org/abs/2403.11134), ArXiv Sun, 17 Mar 2024
- [3D Gaussian as a New Vision Era: A Survey](https://arxiv.org/abs/2402.07181), ArXiv Sun, 11 Feb 2024
- [A Survey on 3D Gaussian Splatting](https://arxiv.org/pdf/2401.03890.pdf), ArXiv 2024
  
#### Awesome Repos
- Resource1: [Awesome 3D Gaussian Splatting Resources](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
- Resource2: [3D Gaussian Splatting Papers](https://github.com/Awesome3DGS/3D-Gaussian-Splatting-Papers)
- Resource3: [3DGS and Beyond Docs](https://github.com/yangjiheng/3DGS_and_Beyond_Docs)

</details>

<details close>
<summary>🔥 Topic 2: AIGC 3D </summary>
 
#### Survey
- [Advances in 3D Generation: A Survey](https://arxiv.org/abs/2401.17807), ArXiv 2024
- [A Comprehensive Survey on 3D Content Generation](https://arxiv.org/abs/2402.01166), ArXiv 2024
- [A Survey On Text-to-3D Contents Generation In The Wild](https://arxiv.org/pdf/2405.09431), ArXiv 2024

#### Awesome Repos
- Resource1: [Awesome 3D AIGC 1](https://github.com/mdyao/Awesome-3D-AIGC) and [Awesome 3D AIGC 2](https://github.com/hitcslj/Awesome-AIGC-3D)
- Resource2: [Awesome Text 2 3D](https://github.com/StellarCheng/Awesome-Text-to-3D)

#### Benchmars
- text-to-3d generation: [GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation](https://arxiv.org/abs/2401.04092), Wu et al., arXiv 2024 | [Code](https://github.com/3DTopia/GPTEval3D)
</details>

<details close>
<summary>🔥 Topic 3: LLM 3D </summary>
 
#### Awesome Repos
- Resource1: [Awesome LLM 3D](https://github.com/ActiveVisionLab/Awesome-LLM-3D)


#### 3D Human
- Survey: [PROGRESS AND PROSPECTS IN 3D GENERATIVE AI: A TECHNICAL OVERVIEW INCLUDING 3D HUMAN](https://arxiv.org/pdf/2401.02620.pdf), ArXiv 2024
- Survey: [A Survey on 3D Human Avatar Modeling -- From Reconstruction to Generation](https://arxiv.org/abs/2406.04253), ArXiv 6 June 2024
- Resource1: [Awesome Digital Human](https://github.com/weihaox/awesome-digital-human)
- Resource2: [Awesome-Avatars](https://github.com/pansanity666/Awesome-Avatars)

</details>

<details close>
<summary>🔥 Topic 4: AIGC 4D </summary>
	
#### Awesome Repos
- Resource1: [Awesome 4D Generation](https://github.com/cwchenwang/awesome-4d-generation)

</details>

<details close>
<summary>Dynamic Gaussian Splatting</summary>
<details close>
<summary>Neural Deformable 3D Gaussians</summary>
 
(CVPR 2024) Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction [Paper](https://arxiv.org/abs/2309.13101) [Code](https://github.com/ingra14m/Deformable-3D-Gaussians) [Page](https://ingra14m.github.io/Deformable-Gaussians/)
 
(CVPR 2024) 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering [Paper](https://arxiv.org/abs/2310.08528) [Code](https://github.com/hustvl/4DGaussians) [Page](https://guanjunwu.github.io/4dgs/index.html)

(CVPR 2024) SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes [Paper](https://arxiv.org/abs/2312.14937) [Code](https://github.com/yihua7/SC-GS) [Page](https://yihua7.github.io/SC-GS-web/)

(CVPR 2024, Highlight) 3DGStream: On-the-Fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos [Paper](https://arxiv.org/abs/2403.01444) [Code](https://github.com/SJoJoK/3DGStream) [Page](https://sjojok.github.io/3dgstream/)

</details>

<details close>
<summary>4D Gaussians</summary>

(ArXiv 2024.02.07) 4D Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes [Paper](https://arxiv.org/abs/2402.03307)
 
(ICLR 2024) Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting [Paper](https://arxiv.org/abs/2310.10642) [Code](https://github.com/fudan-zvg/4d-gaussian-splatting) [Page](https://fudan-zvg.github.io/4d-gaussian-splatting/)

</details>

<details close>
<summary>Dynamic 3D Gaussians</summary>

(CVPR 2024) Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle [Paper](https://arxiv.org/abs/2312.03431) [Page](https://nju-3dv.github.io/projects/Gaussian-Flow/)
 
(3DV 2024) Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis [Paper](https://arxiv.org/abs/2308.09713) [Code](https://github.com/JonathonLuiten/Dynamic3DGaussians) [Page](https://dynamic3dgaussians.github.io/)

</details>

</details>

--------------

## License 
This repo is released under the [MIT license](./LICENSE).

✉️ Any additions or suggestions, feel free to contact us. 
