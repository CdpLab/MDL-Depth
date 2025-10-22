<div id="top" align="center">
  
# MDL-Depth 
**Multi-frame-based Dynamic Scene Lightweight Self-Supervised Monocular Depth Estimation**
  
  Jia Liu, Guorui Lu, Yiyang Wang, Lina Wei, Dapeng Chen*
  
</div>

## Abstract
To address the challenge of self-supervised monocular depth estimation experiencing severe performance degradation in real-world dynamic scenes due to violating the "static world assumption", this paper proposes a unified lightweight single-frame and multi-frame fusion architecture (MDL-Depth). To effectively mitigate interference from moving objects, the method is designed at both the data and loss levels. First, we incorporate a optical flow network to precisely capture pixel-level motion. Through explicit feature warping operations, reference frame features containing dynamic objects are aligned to the target frame's viewpoint, effectively compensating for the independent motion of objects prior to feature fusion. Second, we develop an adaptive loss masking strategy tailored for dynamic foregrounds. By analyzing reprojection errors between consecutive frames, we generate dynamic masks that actively identify and filter out erroneous gradient signals caused by motion mismatches. Experiments on challenging Cityscapes and KITTI datasets demonstrate that our method achieves high accuracy (AbsRel 0.91) in both dynamic and static regions. While ensuring high performance, the framework also prioritizes lightweight design. This provides an effective solution for efficient and precise depth estimation in complex dynamic environments.
## Overview
<"./img/Figure_1.pdf" width="100%" alt="overview" align=center />

## Comparison of KITTI dataset visualizations
<"./img/Figure_2.pdf" width="100%" alt="overview" align=center />

## Comparison of KITTI dataset results 
| Model                                | Parameters (M) | AbsRel | SqRel | RMSE  | RMSElog | δ1   | δ2   | δ3   |
|--------------------------------------|----------------|--------|-------|-------|---------|-------|-------|-------|
| Zhou                                 | 34.2           | 0.208  | 1.768 | 6.958 | 0.283   | 0.678 | 0.885 | 0.957 |
| SGDepth                              | 16.3           | 0.113  | 0.835 | 4.693 | 0.191   | 0.879 | 0.961 | 0.981 |
| MonoFormer-ViT                       | 23.9           | 0.108  | 0.960 | 4.594 | 0.184   | 0.884 | 0.950 | 0.981 |
| Monodepth2                           | 32.5           | 0.115  | 0.903 | 4.863 | 0.193   | 0.877 | 0.959 | 0.981 |
| R-MSMF6                              | 3.8            | 0.120  | 1.062 | 5.800 | 0.204   | 0.857 | 0.948 | 0.978 |
| Lite-Mono                            | 3.1            | 0.107  | 0.765 | 4.461 | 0.183   | 0.886 | 0.960 | 0.979 |
| MonoViT-tiny                         | 10.3           | 0.106  | 0.749 | 4.484 | 0.183   | 0.888 | 0.961 | 0.980 |
| HR-Depth                             | 14.7           | 0.109  | 0.792 | 4.632 | 0.185   | 0.884 | 0.959 | 0.979 |
| Sc-depth3                            | 59.3           | 0.118  | 0.756 | 4.756 | 0.188   | 0.844 | 0.960 | 0.980 |
| DNA-Depth-B0                         | 9.1            | 0.130  | 1.053 | 5.144 | 0.208   | 0.853 | 0.940 | 0.979 |
| Bian                                 | 7.0            | 0.125  | 0.856 | 5.071 | 0.201   | 0.849 | 0.948 | 0.980 |
| **Ours**                             | **3.0**        | **0.102** | **0.746** | **4.543** | **0.178** | **0.896** | **0.964** | **0.983** |



## Data Preparation
Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to prepare your KITTI data.

## Install

The models were trained using CUDA 11.8, Python 3.9.x (conda environment), and PyTorch 2.4.1.

Create a conda environment with the PyTorch library:

```bash
conda create -n LSMDepth python=3.9.4
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate LSMDepth
```

Install prerequisite packages listed in requirements.txt:
```bash
pip install -r requirements.txt
```

## Training
The models can be trained on the KITTI dataset by running:
```bash
python train.py --data_path path/to/your/data --model_name mymodel
```

## Inference
To inference on a single image,run:
```bash
python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image
```
## Evaluation
To evaluate a model on KITTI, run:
```bash
python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --data_path path/to/kitti_data/ --model lite-mono
```
