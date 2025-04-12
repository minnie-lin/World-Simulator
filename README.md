# Simulating the Real World: Survey & Resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-pink.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-pink) [![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-pink)]()
![Stars](https://img.shields.io/github/stars/ALEEEHU/World-Simulator)

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
  - [Other Related Resources](#other-related-resources)
    * [World Foundation Model Platform](#world-foundation-model-platform)
- [Text2X Resources](#-awesome-text2x-resources)
  - [Text to 4D](#text-to-4d)
    * [Accepted Papers](#-4d-accepted-papers)
    * [ArXiv Papers](#-4d-arxiv-papers)
  - [Text to Video](#text-to-video)
    * [Accepted Papers](#-t2v-accepted-papers)
    * [ArXiv Papers](#-t2v-arxiv-papers)
    * [Additional Info](#video-other-additional-info)
  - [Text to 3D Scene](#text-to-scene)
    * [ArXiv Papers](#-3d-scene-arxiv-papers)
  - [Text to Human Motion](#text-to-human-motion)
    * [Accepted Papers](#-motion-accepted-papers)
    * [ArXiv Papers](#-motion-arxiv-papers)
    * [Additional Info](#motion-other-additional-info)
  - [Text to 3D Human](#text-to-3d-human)
    * [Accepted Papers](#-human-accepted-papers)
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

@article{hu2025simulating,
  title={Simulating the Real World: A Unified Survey of Multimodal Generative Models},
  author={Hu, Yuqi and Wang, Longguang and Liu, Xian and Chen, Ling-Hao and Guo, Yuwei and Shi, Yukai and Liu, Ce and Rao, Anyi and Wang, Zeyu and Xiong, Hui},
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
* [ICLR 2025] **IPDreamer: Appearance-Controllable 3D Object Generation with Complex Image Prompts** [[Paper](https://arxiv.org/pdf/2310.05375)] [[GitHub](https://github.com/zengbohan0217/IPDreamer)]

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
* [arXiv 2024] **Trans4D: Realistic Geometry-Aware Transition for Compositional Text-to-4D Synthesis** [[Paper](https://arxiv.org/pdf/2410.07155)] [[GitHub](https://github.com/YangLing0818/Trans4D)]

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



## Other Related Resources

### World Foundation Model Platform
- [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) ([[GitHub](https://github.com/nvidia-cosmos)] [[Paper](https://arxiv.org/abs/2501.03575)]): NVIDIA Cosmos is a world foundation model platform for accelerating the development of physical AI systems.
  
	- [Cosmos-Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1)：a world-to-world transfer model designed to bridge the perceptual divide between simulated and real-world environments.
   	- [Cosmos-Predict1](https://github.com/nvidia-cosmos/cosmos-predict1): a collection of general-purpose world foundation models for Physical AI that can be fine-tuned into customized world models for downstream applications.
   	- [Cosmos-Reason1](https://github.com/nvidia-cosmos/cosmos-reason1)： a model that understands the physical common sense and generate appropriate embodied decisions in natural language through long chain-of-thought reasoning processes.

-------

[<u>🎯Back to Top - Our Survey Paper Collection</u>](#-our-survey-paper-collection)

## 🔥 Awesome Text2X Resources

An open collection of state-of-the-art (SOTA), novel **Text to X (X can be everything)** methods (papers, codes and datasets), intended to keep pace with the anticipated surge of research.

<div><div align="center">
	<img width="500" height="350" src="media/logo.svg" alt="Awesome"></div>



## Update Logs
* `2025.03.10` - [CVPR 2025 Accepted Papers](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)🎉
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

### 🎉 4D Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2501.02690)          | [Link](https://github.com/wkbian/GS-DiT)  | [Link](https://wkbian.github.io/Projects/GS-DiT/)  |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@article{bian2025gsdit,
  title={GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking},
  author={Bian, Weikang and Huang, Zhaoyang and Shi, Xiaoyu and and Li, Yijin and Wang, Fu-Yun and Li, Hongsheng},
  journal={arXiv preprint arXiv:2501.02690},
  year={2025}
}
```
</details>

-------

### 💡 4D ArXiv Papers

#### 1. AR4D: Autoregressive 4D Generation from Monocular Videos
Hanxin Zhu, Tianyu He, Xiqian Yu, Junliang Guo, Zhibo Chen, Jiang Bian (University of Science and Technology of China, Microsoft Research Asia)
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in generative models have ignited substantial interest in dynamic 3D content creation (\ie, 4D generation). Existing approaches primarily rely on Score Distillation Sampling (SDS) to infer novel-view videos, typically leading to issues such as limited diversity, spatial-temporal inconsistency and poor prompt alignment, due to the inherent randomness of SDS. To tackle these problems, we propose AR4D, a novel paradigm for SDS-free 4D generation. Specifically, our paradigm consists of three stages. To begin with, for a monocular video that is either generated or captured, we first utilize pre-trained expert models to create a 3D representation of the first frame, which is further fine-tuned to serve as the canonical space. Subsequently, motivated by the fact that videos happen naturally in an autoregressive manner, we propose to generate each frame's 3D representation based on its previous frame's representation, as this autoregressive generation manner can facilitate more accurate geometry and motion estimation. Meanwhile, to prevent overfitting during this process, we introduce a progressive view sampling strategy, utilizing priors from pre-trained large-scale 3D reconstruction models. To avoid appearance drift introduced by autoregressive generation, we further incorporate a refinement stage based on a global deformation field and the geometry of each frame's 3D representation. Extensive experiments have demonstrated that AR4D can achieve state-of-the-art 4D generation without SDS, delivering greater diversity, improved spatial-temporal consistency, and better alignment with input prompts.
</details>

#### 2. WideRange4D: Enabling High-Quality 4D Reconstruction with Wide-Range Movements and Scenes
Ling Yang, Kaixin Zhu, Juanxi Tian, Bohan Zeng, Mingbao Lin, Hongjuan Pei, Wentao Zhang, Shuicheng Yan 

(Peking University, University of the Chinese Academy of Sciences, National University of Singapore)
<details span>
<summary><b>Abstract</b></summary>
With the rapid development of 3D reconstruction technology, research in 4D reconstruction is also advancing, existing 4D reconstruction methods can generate high-quality 4D scenes. However, due to the challenges in acquiring multi-view video data, the current 4D reconstruction benchmarks mainly display actions performed in place, such as dancing, within limited scenarios. In practical scenarios, many scenes involve wide-range spatial movements, highlighting the limitations of existing 4D reconstruction datasets. Additionally, existing 4D reconstruction methods rely on deformation fields to estimate the dynamics of 3D objects, but deformation fields struggle with wide-range spatial movements, which limits the ability to achieve high-quality 4D scene reconstruction with wide-range spatial movements. In this paper, we focus on 4D scene reconstruction with significant object spatial movements and propose a novel 4D reconstruction benchmark, WideRange4D. This benchmark includes rich 4D scene data with large spatial variations, allowing for a more comprehensive evaluation of the generation capabilities of 4D generation methods. Furthermore, we introduce a new 4D reconstruction method, Progress4D, which generates stable and high-quality 4D results across various complex 4D scene reconstruction tasks. We conduct both quantitative and qualitative comparison experiments on WideRange4D, showing that our Progress4D outperforms existing state-of-the-art 4D reconstruction methods. 
</details>

#### 3. SV4D 2.0: Enhancing Spatio-Temporal Consistency in Multi-View Video Diffusion for High-Quality 4D Generation
Chun-Han Yao, Yiming Xie, Vikram Voleti, Huaizu Jiang, Varun Jampani 

(Stability AI, Northeastern University)
<details span>
<summary><b>Abstract</b></summary>
We present Stable Video 4D 2.0 (SV4D 2.0), a multi-view video diffusion model for dynamic 3D asset generation. Compared to its predecessor SV4D, SV4D 2.0 is more robust to occlusions and large motion, generalizes better to real-world videos, and produces higher-quality outputs in terms of detail sharpness and spatio-temporal consistency. We achieve this by introducing key improvements in multiple aspects: 1) network architecture: eliminating the dependency of reference multi-views and designing blending mechanism for 3D and frame attention, 2) data: enhancing quality and quantity of training data, 3) training strategy: adopting progressive 3D-4D training for better generalization, and 4) 4D optimization: handling 3D inconsistency and large motion via 2-stage refinement and progressive frame sampling. Extensive experiments demonstrate significant performance gain by SV4D 2.0 both visually and quantitatively, achieving better detail (-14\% LPIPS) and 4D consistency (-44\% FV4D) in novel-view video synthesis and 4D optimization (-12\% LPIPS and -24\% FV4D) compared to SV4D. 
</details>

#### 4. Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency
Tianqi Liu, Zihao Huang, Zhaoxi Chen, Guangcong Wang, Shoukang Hu, Liao Shen, Huiqiang Sun, Zhiguo Cao, Wei Li, Ziwei Liu

(Huazhong University of Science and Technology, Nanyang Technological University, Great Bay University)
<details span>
<summary><b>Abstract</b></summary>
We present Free4D, a novel tuning-free framework for 4D scene generation from a single image. Existing methods either focus on object-level generation, making scene-level generation infeasible, or rely on large-scale multi-view video datasets for expensive training, with limited generalization ability due to the scarcity of 4D scene data. In contrast, our key insight is to distill pre-trained foundation models for consistent 4D scene representation, which offers promising advantages such as efficiency and generalizability. 1) To achieve this, we first animate the input image using image-to-video diffusion models followed by 4D geometric structure initialization. 2) To turn this coarse structure into spatial-temporal consistent multiview videos, we design an adaptive guidance mechanism with a point-guided denoising strategy for spatial consistency and a novel latent replacement strategy for temporal coherence. 3) To lift these generated observations into consistent 4D representation, we propose a modulation-based refinement to mitigate inconsistencies while fully leveraging the generated information. The resulting 4D representation enables real-time, controllable rendering, marking a significant advancement in single-image-based 4D scene generation.
</details>

-----

</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **AR4D: Autoregressive 4D Generation from Monocular Videos**  | 3 Jan 2025 |          [Link](https://arxiv.org/abs/2501.01722)          | --  | [Link](https://hanxinzhu-lab.github.io/AR4D/)  |
| 2025 | **WideRange4D: Enabling High-Quality 4D Reconstruction with Wide-Range Movements and Scenes**  | 17 Mar 2025 |          [Link](https://arxiv.org/abs/2503.13435)          | [Link](https://github.com/Gen-Verse/WideRange4D)  | [Dataset Page](https://huggingface.co/datasets/Gen-Verse/WideRange4D)  |
| 2025 | **SV4D 2.0: Enhancing Spatio-Temporal Consistency in Multi-View Video Diffusion for High-Quality 4D Generation**  | 20 Mar 2025 |          [Link](https://arxiv.org/abs/2503.16396)          | --  | [Link](https://sv4d2.0.github.io/)  |
| 2025 | **Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency**  | 26 Mar 2025 |          [Link](https://arxiv.org/abs/2503.20785)          | [Link](https://github.com/TQTQliu/Free4D)  | [Link](https://free4d.github.io/)  |

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

@article{yang2025widerange4d,
  title={WideRange4D: Enabling High-Quality 4D Reconstruction with Wide-Range Movements and Scenes},
  author={Yang, Ling and Zhu, Kaixin and Tian, Juanxi and Zeng, Bohan and Lin, Mingbao and Pei, Hongjuan and Zhang, Wentao and Yan, Shuichen},
  journal={arXiv preprint arXiv:2503.13435},
  year={2025}
}

@misc{yao2025sv4d20enhancingspatiotemporal,
      title={SV4D 2.0: Enhancing Spatio-Temporal Consistency in Multi-View Video Diffusion for High-Quality 4D Generation}, 
      author={Chun-Han Yao and Yiming Xie and Vikram Voleti and Huaizu Jiang and Varun Jampani},
      year={2025},
      eprint={2503.16396},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.16396}, 
}

 @article{liu2025free4d,
     title={Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency},
     author={Liu, Tianqi and Huang, Zihao and Chen, Zhaoxi and Wang, Guangcong and Hu, Shoukang and Shen, liao and Sun, Huiqiang and Cao, Zhiguo and Li, Wei and Liu, Ziwei},
     journal={arXiv preprint arXiv:2503.20785},
     year={2025}
 }
```
</details>

---

### Previous Papers

### Year 2023
In 2023, tasks classified as text/Image to 4D and video to 4D generally involve producing four-dimensional data from text/Image or video input. For more details, please check the [2023 4D Papers](./docs/4d/4d_2023.md), including 6 accepted papers and 3 arXiv papers.

### Year 2024
For more details, please check the [2024 4D Papers](./docs/4d/4d_2024.md), including 21 accepted papers and 13 arXiv papers.

--------------


## Text to Video

### 🎉 T2V Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **TransPixar: Advancing Text-to-Video Generation with Transparency**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2501.03006)          | [Link](https://github.com/wileewang/TransPixar)  | [Link](https://wileewang.github.io/TransPixar/)  |
| 2025 | **BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2501.07647)          | -- | [Link](https://blobgen-vid2.github.io/)  |
| 2025 | **Identity-Preserving Text-to-Video Generation by Frequency Decomposition**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2411.17440)          | [Link](https://github.com/PKU-YuanGroup/ConsisID) | [Link](https://pku-yuangroup.github.io/ConsisID/)  |
| 2025 | **One-Minute Video Generation with Test-Time Training**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2504.05298v1)          | [Link](https://github.com/test-time-training/ttt-video-dit) | [Link](https://test-time-training.github.io/video-dit/)  |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@misc{wang2025transpixar,
     title={TransPixar: Advancing Text-to-Video Generation with Transparency}, 
     author={Luozhou Wang and Yijun Li and Zhifei Chen and Jui-Hsien Wang and Zhifei Zhang and He Zhang and Zhe Lin and Yingcong Chen},
     year={2025},
     eprint={2501.03006},
     archivePrefix={arXiv},
     primaryClass={cs.CV},
     url={https://arxiv.org/abs/2501.03006}, 
}

@article{feng2025blobgen,
  title={BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations},
  author={Feng, Weixi and Liu, Chao and Liu, Sifei and Wang, William Yang and Vahdat, Arash and Nie, Weili},
  journal={arXiv preprint arXiv:2501.07647},
  year={2025}
}

@article{yuan2024identity,
  title={Identity-Preserving Text-to-Video Generation by Frequency Decomposition},
  author={Yuan, Shenghai and Huang, Jinfa and He, Xianyi and Ge, Yunyuan and Shi, Yujun and Chen, Liuhan and Luo, Jiebo and Yuan, Li},
  journal={arXiv preprint arXiv:2411.17440},
  year={2024}
}

@misc{dalal2025oneminutevideogenerationtesttime,
      title={One-Minute Video Generation with Test-Time Training}, 
      author={Karan Dalal and Daniel Koceja and Gashon Hussein and Jiarui Xu and Yue Zhao and Youjin Song and Shihao Han and Ka Chun Cheung and Jan Kautz and Carlos Guestrin and Tatsunori Hashimoto and Sanmi Koyejo and Yejin Choi and Yu Sun and Xiaolong Wang},
      year={2025},
      eprint={2504.05298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.05298}, 
}

```
</details>

-------

### 💡 T2V ArXiv Papers

#### 1. Multi-subject Open-set Personalization in Video Generation
Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, Sergey Tulyakov

(Snap Inc., UC Merced, CMU)
<details span>
<summary><b>Abstract</b></summary>
Video personalization methods allow us to synthesize videos with specific concepts such as people, pets, and places. However, existing methods often focus on limited domains, require time-consuming optimization per subject, or support only a single subject. We present Video Alchemist − a video model with built-in multi-subject, open-set personalization capabilities for both foreground objects and background, eliminating the need for time-consuming test-time optimization. Our model is built on a new Diffusion Transformer module that fuses each conditional reference image and its corresponding subject-level text prompt with cross-attention layers. Developing such a large model presents two main challenges: dataset and evaluation. First, as paired datasets of reference images and videos are extremely hard to collect, we sample selected video frames as reference images and synthesize a clip of the target video. However, while models can easily denoise training videos given reference frames, they fail to generalize to new contexts. To mitigate this issue, we design a new automatic data construction pipeline with extensive image augmentations. Second, evaluating open-set video personalization is a challenge in itself. To address this, we introduce a personalization benchmark that focuses on accurate subject fidelity and supports diverse personalization scenarios. Finally, our extensive experiments show that our method significantly outperforms existing personalization methods in both quantitative and qualitative evaluations.
</details>


| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **Multi-subject Open-set Personalization in Video Generation**  | 10 Jan 2025 |          [Link](https://arxiv.org/abs/2501.06187)          | -- | [Link](https://snap-research.github.io/open-set-video-personalization/)  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@misc{chen2025multisubjectopensetpersonalizationvideo,
      title={Multi-subject Open-set Personalization in Video Generation}, 
      author={Tsai-Shien Chen and Aliaksandr Siarohin and Willi Menapace and Yuwei Fang and Kwot Sin Lee and Ivan Skorokhodov and Kfir Aberman and Jun-Yan Zhu and Ming-Hsuan Yang and Sergey Tulyakov},
      year={2025},
      eprint={2501.06187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.06187}, 
}

```
</details>


---

### Video Other Additional Info

### Previous Papers

### Year 2024
For more details, please check the [2024 T2V Papers](./docs/video/t2v_2024.md), including 21 accepted papers and 6 arXiv papers.

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

#### 2. Bolt3D: Generating 3D Scenes in Seconds
Stanislaw Szymanowicz, Jason Y. Zhang, Pratul Srinivasan, Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo Martin-Brualla, Jonathan T. Barron, Philipp Henzler 

(Google Research, University of Oxford, Google DeepMind)
<details span>
<summary><b>Abstract</b></summary>
We present a latent diffusion model for fast feed-forward 3D scene generation. Given one or more images, our model Bolt3D directly samples a 3D scene representation in less than seven seconds on a single GPU. We achieve this by leveraging powerful and scalable existing 2D diffusion network architectures to produce consistent high-fidelity 3D scene representations. To train this model, we create a large-scale multiview-consistent dataset of 3D geometry and appearance by applying state-of-the-art dense 3D reconstruction techniques to existing multiview image datasets. Compared to prior multiview generative models that require per-scene optimization for 3D reconstruction, Bolt3D reduces the inference cost by a factor of up to 300 times.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation**  | 4 Feb 2025 |          [Link](https://arxiv.org/abs/2502.01949)          | --  | --  |
| 2025 | **Bolt3D: Generating 3D Scenes in Seconds**  | 18 Mar 2025 |          [Link](https://arxiv.org/abs/2503.14445)          | --  | [Link](https://szymanowiczs.github.io/bolt3d)  |

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

@article{szymanowicz2025bolt3d,
title={{Bolt3D: Generating 3D Scenes in Seconds}},
author={Szymanowicz, Stanislaw and Zhang, Jason Y. and Srinivasan, Pratul
     and Gao, Ruiqi and Brussee, Arthur and Holynski, Aleksander and
     Martin-Brualla, Ricardo and Barron, Jonathan T. and Henzler, Philipp},
journal={arXiv:2503.14445},
year={2025}
}

```
</details>

### Previous Papers

### Year 2023-2024
For more details, please check the [2023-2024 3D Scene Papers](./docs/3d_scene/3d_scene_23-24.md), including 22 accepted papers and 9 arXiv papers.

--------------


## Text to Human Motion

### 🎉 Motion Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **MixerMDM: Learnable Composition of Human Motion Diffusion Models**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2504.01019)          | [Link](https://github.com/pabloruizponce/MixerMDM)  | [Link](https://www.pabloruizponce.com/papers/MixerMDM)  |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@article{ruiz2025mixermdm,
  title={MixerMDM: Learnable Composition of Human Motion Diffusion Models},
  author={Ruiz-Ponce, Pablo and Barquero, German and Palmero, Cristina and Escalera, Sergio and Garc{\'\i}a-Rodr{\'\i}guez, Jos{\'e}},
  journal={arXiv preprint arXiv:2504.01019},
  year={2025}
}

```
</details>

-------

### 💡 Motion ArXiv Papers

#### 1. MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm
Ziyan Guo, Zeyu Hu, Na Zhao, De Wen Soh 

(Singapore University of Technology and Design, LightSpeed Studios)
<details span>
<summary><b>Abstract</b></summary>
Human motion generation and editing are key components of computer graphics and vision. However, current approaches in this field tend to offer isolated solutions tailored to specific tasks, which can be inefficient and impractical for real-world applications. While some efforts have aimed to unify motion-related tasks, these methods simply use different modalities as conditions to guide motion generation. Consequently, they lack editing capabilities, fine-grained control, and fail to facilitate knowledge sharing across tasks. To address these limitations and provide a versatile, unified framework capable of handling both human motion generation and editing, we introduce a novel paradigm: Motion-Condition-Motion, which enables the unified formulation of diverse tasks with three concepts: source motion, condition, and target motion. Based on this paradigm, we propose a unified framework, MotionLab, which incorporates rectified flows to learn the mapping from source motion to target motion, guided by the specified conditions. In MotionLab, we introduce the 1) MotionFlow Transformer to enhance conditional generation and editing without task-specific modules; 2) Aligned Rotational Position Encoding} to guarantee the time synchronization between source motion and target motion; 3) Task Specified Instruction Modulation; and 4) Motion Curriculum Learning for effective multi-task learning and knowledge sharing across tasks. Notably, our MotionLab demonstrates promising generalization capabilities and inference efficiency across multiple benchmarks for human motion.
</details>

#### 2. Motion Anything: Any to Motion Generation
Zeyu Zhang, Yiran Wang, Wei Mao, Danning Li, Rui Zhao, Biao Wu, Zirui Song, Bohan Zhuang, Ian Reid, Richard Hartley

(The Australian National University, The University of Sydney, Tecent Canberra XR Vision Labs, McGill University, JD.com, University of Technology Sydney, Mohamed bin Zayed University of Artificial Intelligence, Zhejiang University, Google Research)
<details span>
<summary><b>Abstract</b></summary>
Conditional motion generation has been extensively studied in computer vision, yet two critical challenges remain. First, while masked autoregressive methods have recently outperformed diffusion-based approaches, existing masking models lack a mechanism to prioritize dynamic frames and body parts based on given conditions. Second, existing methods for different conditioning modalities often fail to integrate multiple modalities effectively, limiting control and coherence in generated motion. To address these challenges, we propose Motion Anything, a multimodal motion generation framework that introduces an Attention-based Mask Modeling approach, enabling fine-grained spatial and temporal control over key frames and actions. Our model adaptively encodes multimodal conditions, including text and music, improving controllability. Additionally, we introduce Text-Music-Dance (TMD), a new motion dataset consisting of 2,153 pairs of text, music, and dance, making it twice the size of AIST++, thereby filling a critical gap in the community. Extensive experiments demonstrate that Motion Anything surpasses state-of-the-art methods across multiple benchmarks, achieving a 15% improvement in FID on HumanML3D and showing consistent performance gains on AIST++ and TMD. 
</details>

#### 3. MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space
Lixing Xiao, Shunlin Lu, Huaijin Pi, Ke Fan, Liang Pan, Yueer Zhou, Ziyong Feng, Xiaowei Zhou, Sida Peng, Jingbo Wang

(Zhejiang University, The Chinese University of Hong Kong Shenzhen, The University of Hong Kong, Shanghai Jiao Tong University, DeepGlint, Shanghai AI Laboratory)
<details span>
<summary><b>Abstract</b></summary>
This paper addresses the challenge of text-conditioned streaming motion generation, which requires us to predict the next-step human pose based on variable-length historical motions and incoming texts. Existing methods struggle to achieve streaming motion generation, e.g., diffusion models are constrained by pre-defined motion lengths, while GPT-based methods suffer from delayed response and error accumulation problem due to discretized non-causal tokenization. To solve these problems, we propose MotionStreamer, a novel framework that incorporates a continuous causal latent space into a probabilistic autoregressive model. The continuous latents mitigate information loss caused by discretization and effectively reduce error accumulation during long-term autoregressive generation. In addition, by establishing temporal causal dependencies between current and historical motion latents, our model fully utilizes the available information to achieve accurate online motion decoding. Experiments show that our method outperforms existing approaches while offering more applications, including multi-round generation, long-term generation, and dynamic motion composition. 
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm**  | 6 Feb 2025 |          [Link](https://arxiv.org/abs/2502.02358)          | [Link](https://github.com/Diouo/MotionLab)  | [Link](https://diouo.github.io/motionlab.github.io/)  |
| 2025 | **Motion Anything: Any to Motion Generation**  | 12 Mar 2025 |          [Link](https://arxiv.org/abs/2503.06955)          | [Link](https://github.com/steve-zeyu-zhang/MotionAnything)  | [Link](https://steve-zeyu-zhang.github.io/MotionAnything/)  |
| 2025 | **MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space**  | 19 Mar 2025 |          [Link](https://arxiv.org/abs/2503.15451)          | [Link](https://github.com/zju3dv/MotionStreamer)  | [Link](https://zju3dv.github.io/MotionStreamer/)  |

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

@article{zhang2025motion,
  title={Motion Anything: Any to Motion Generation},
  author={Zhang, Zeyu and Wang, Yiran and Mao, Wei and Li, Danning and Zhao, Rui and Wu, Biao and Song, Zirui and Zhuang, Bohan and Reid, Ian and Hartley, Richard},
  journal={arXiv preprint arXiv:2503.06955},
  year={2025}
}

@article{xiao2025motionstreamer,
      title={MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space},
      author={Xiao, Lixing and Lu, Shunlin and Pi, Huaijin and Fan, Ke and Pan, Liang and Zhou, Yueer and Feng, Ziyong and Zhou, Xiaowei and Peng, Sida and Wang, Jingbo},
      journal={arXiv preprint arXiv:2503.15451},
      year={2025}
}
```
</details>


---

### Motion Other Additional Info

### Previous Papers

### Year 2023-2024
For more details, please check the [2023-2024 Text to Human Motion Papers](./docs/human_motion/motion_23-24.md), including 36 accepted papers and 8 arXiv papers.

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

### 🎉 Human Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2503.15851)          | [Link](https://github.com/ZhenglinZhou/Zero-1-to-A)  | [Link](https://zhenglinzhou.github.io/Zero-1-to-A/)  |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@inproceedings{zhou2025zero1toa,
  title = {Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion},
  author = {Zhenglin Zhou and Fan Ma and Hehe Fan and Tat-Seng Chua},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}

```
</details>


---------

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
For more details, please check the [2023-2024 3D Human Papers](./docs/3d_human/human_23-24.md), including 18 accepted papers and 5 arXiv papers.

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

[<u>🎯Back to Top - Text2X Resources</u>](#-awesome-text2x-resources)


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

#### Foundation Model
- [Cube](https://github.com/Roblox/cube), [ArXiv Report](https://arxiv.org/abs/2503.15475)
 
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

#### Survey
- [Advances in 4D Generation: A Survey](https://arxiv.org/abs/2503.14501), ArXiv 2025
	
#### Awesome Repos
- Resource1: [Awesome 4D Generation](https://github.com/cwchenwang/awesome-4d-generation)

</details>

<details close>
<summary>🔥 Topic 5: Physics-based AIGC </summary>

#### Survey
- [Exploring the Evolution of Physics Cognition in Video Generation: A Survey](https://arxiv.org/abs/2503.21765), ArXiv 2025
	
#### Awesome Repos
- Resource1: [Awesome-Physics-Cognition-based-Video-Generation](https://github.com/minnie-lin/Awesome-Physics-Cognition-based-Video-Generation)

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

[<u>🎯Back to Top - Table of Contents</u>](#table-of-contents)


## License 
This repo is released under the [MIT license](./LICENSE).

✉️ Any additions or suggestions, feel free to contact us. 
