# AIM2020-RealSR
Our solution to AIM2020 Real Image Super-Resolution Challenge (x2). **x2 SSIM Rank 3rd** at the end of the Development phase (2020.7.10). **We propose a new "[crop-ensemble](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/README.md#51-crop-ensemble)" and it is compatible with model-ensemble and self-ensemble to achieve higher performances.**

## 1. Basic Models :
Our solution consists of three basic models (**model ensemble**): OADDetv1, OADDetv2 and Deep-OADDet. OADDetv1 and v2 shares the same architecture yet trained on different datasets (further details in [Training Scripts](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/README.md#33-training-scripts) and [Dataset](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/README.md#dataset)). 

<img src="./img/OADDet.jpg" width="500" height="250" />

Our core modules are heavily borrowed from [DDet](https://github.com/ykshi/DDet), [Inception](https://arxiv.org/pdf/1409.4842.pdf) and [OANet](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Du_Orientation-Aware_Deep_Neural_Network_for_Real_Image_Super-Resolution_CVPRW_2019_paper.pdf) with minor improvements, such as fewer attention modules, skip connections and LeakyReLU.

<img src="./img/OADDet_Network.jpg" width="640" height="360" />

## 2. Environment :
We conduct all experiments on Nvidia GPUs (NVIDIA Tesla V100 SXM2 16GB) including training (12 GPUs) and testing (4 GPUs). The total training time is about 2000 GPU hours on V100. It takes about 30GB DRAM during training. The detailed requirements are as follow:
```C
DRAM>=32GB
Pillow==6.0.0
GCC==7.5.0
python==3.7.3
torch==1.1.0
torchvision==0.2.2.post3
CUDA==9.0.176
imageio==2.5.0
numpy==1.16.3
opencv-contrib-python==4.3.0.36
scikit-image==0.15.0
scipy==1.2.1
```

## 3. How to use ?
### 3.1 Reproduce x2 test dataset results:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model WDDet --n_resblocks 40 --n_feats 128 --res_scale 1.0 --data_test AIM --scale 2 --save AIM_WDDet_x2_VAL_model_latest --test_only --dir_data /nfsdata1/home/hexiangyu/RealSR_X2_Full_Valid_New/ --pre_train /nfsdata1/home/hexiangyu/EDSR-PyTorch-legacy-1.1.0/experiment/AIM_WDDet_x2_Large_Dataset_SSIM_Finetune/model/model_1.pt --n_GPUs 1 --chop --chop-size 410 --shave-size 10
```
### 3.2 Test on your own images:
```shell
CUDA_VISIBLE_DEVICES=0,1 python main.py --model DDDet --n_resblocks 32 --n_feats 128 --res_scale 1.0 --data_test Demo --scale 2 --save Demo_x2_ouptut --test_only --save_results --dir_demo /your/image/dir/ --pre_train ../experiment/AIM_DDet_x2_4th_so_far_best_Large_Dataset_SSIM_Finetune/model/model_10.pt --n_GPUs 2 --chop --chop-size 500 --shave-size 100
```

### 3.3 Training Scripts:
We release all our training scripts to help reproduce our results and hopefully, the following methods may benefit from our works.
#### 3.3.1 OADDetv1
Trained on original AIM x2 dataset; Finetuned on washed AIM x2.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model DDDet --scale 2 --save DIV2K_DDet_x2 --n_resblocks 32 --n_feats 128 --res_scale 1.0 --data_train DIV2K --data_test DIV2K --batch_size 32 --dir_data /data/ --ext bin --n_GPUs 4 --reset --patch_size 96 --n_threads 4 --split_batch 1 --lr 1e-4 --decay 100-200 --epochs 300
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model DDDet --scale 2 --save AIM_DDet_x2 --n_resblocks 32 --n_feats 128 --res_scale 1.0 --data_train AIM --data_test AIM --batch_size 32 --dir_data /data/ --ext bin --n_GPUs 4 --reset --patch_size 128 --n_threads 2 --split_batch 1 --lr 5e-5 --decay 150-300-450-600 --epochs 600 --pre_train ../experiment/DIV2K_DDet_x2/model/model_best.pt --save_models --chop
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model DDDet --scale 2 --save AIM_DDet_x2_SSIM_finetune --n_resblocks 32 --n_feats 128 --res_scale 1.0 --data_train AIM --data_test AIM --batch_size 4 --dir_data /data/AIM_washed --ext bin --n_GPUs 4 --reset --patch_size 420 --n_threads 4 --split_batch 1 --lr 1e-6 --decay 100 --epochs 100 --pre_train ../experiment/AIM_DDet_x2/model/model_latest.pt --chop --loss 20.0*SSIM
```
#### 3.3.2 OADDetv2
Trained on washed AIM x2 dataset; Fine-tuned on washed AIM x2+x3 dataset and washed x2 dataset.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model DDDet --scale 2 --save DIV2K_DDet_x2 --n_resblocks 32 --n_feats 128 --res_scale 1.0 --data_train DIV2K --data_test DIV2K --batch_size 32 --dir_data /data/ --ext bin --n_GPUs 4 --reset --patch_size 96 --n_threads 4 --split_batch 1 --lr 1e-4 --decay 30 --epochs 30
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model DDDet --scale 2 --save AIM_DDet_x2 --n_resblocks 32 --n_feats 128 --res_scale 1.0 --data_train AIM --data_test AIM --batch_size 32 --dir_data /data/AIM_washed --ext bin --n_GPUs 4 --reset --patch_size 128 --n_threads 2 --split_batch 1 --lr 5e-5 --decay 100-200-300 --epochs 350 --pre_train ../experiment/DIV2K_DDet_x2/model/model_best.pt --save_models --chop
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model DDDet --scale 2 --save AIM_DDet_x2_L1_finetune --n_resblocks 32 --n_feats 128 --res_scale 1.0 --data_train AIM --data_test AIM --batch_size 32 --dir_data /data/AIM_washed_Large --ext bin --n_GPUs 4 --reset --patch_size 128 --n_threads 4 --split_batch 1 --lr 5e-5 --decay 100-200-300 --epochs 350 --pre_train ../experiment/AIM_DDet_x2/model/model_latest.pt --chop --loss 1.0*L1
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model DDDet --scale 2 --save AIM_DDet_x2_SSIM_finetune --n_resblocks 32 --n_feats 128 --res_scale 1.0 --data_train AIM --data_test AIM --batch_size 4 --dir_data /data/AIM_washed --ext bin --n_GPUs 4 --reset --patch_size 400 --n_threads 4 --split_batch 1 --lr 1e-5 --decay 100 --epochs 100 --pre_train ../experiment/AIM_DDet_x2_L1_finetune/model/model_latest.pt --chop --loss 20.0*SSIM
```
#### 3.3.3 Deep-OADDet
Trained on washed AIM x2 dataset; Fine-tuned on washed AIM x2+x3 dataset and washed x2 dataset.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model WDDet --scale 2 --save DIV2K_WDDet_x2 --n_resblocks 40 --n_feats 128 --res_scale 1.0 --data_train DIV2K --data_test DIV2K --batch_size 32 --dir_data /data/ --ext bin --n_GPUs 4 --reset --patch_size 96 --n_threads 4 --split_batch 1 --lr 1e-4 --decay 30 --epochs 30
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model WDDet --scale 2 --save AIM_WDDet_x2 --n_resblocks 40 --n_feats 128 --res_scale 1.0 --data_train AIM --data_test AIM --batch_size 32 --dir_data /data/AIM_washed --ext bin --n_GPUs 4 --reset --patch_size 128 --n_threads 2 --split_batch 1 --lr 5e-5 --decay 100-200-300 --epochs 350 --pre_train ../experiment/DIV2K_WDDet_x2/model/model_best.pt --save_models --chop
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model WDDet --scale 2 --save AIM_WDDet_x2_L1_finetune --n_resblocks 40 --n_feats 128 --res_scale 1.0 --data_train AIM --data_test AIM --batch_size 32 --dir_data /data/AIM_washed_Large --ext bin --n_GPUs 4 --reset --patch_size 128 --n_threads 4 --split_batch 1 --lr 5e-5 --decay 100-200-300 --epochs 350 --pre_train ../experiment/AIM_WDDet_x2/model/model_latest.pt --chop --loss 1.0*L1
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model WDDet --scale 2 --save AIM_WDDet_x2_SSIM_finetune --n_resblocks 40 --n_feats 128 --res_scale 1.0 --data_train AIM --data_test AIM --batch_size 4 --dir_data /data/AIM_washed --ext bin --n_GPUs 4 --reset --patch_size 400 --n_threads 4 --split_batch 1 --lr 1e-5 --decay 100 --epochs 100 --pre_train ../experiment/AIM_WDDet_x2_L1_finetune/model/model_latest.pt --chop --loss 20.0*SSIM
```

## 4. Dataset :
### 4.1 Washed Dataset :
According to this [issue](https://competitions.codalab.org/forums/21376/3953/):
> I found that many photos in the training dataset are not pixel-wise aligned. Actually, there are different types of misalignment: camera shift, moving objects (e.x. trees, grass).

> However, looking at the dataset, I found that there are very large shifts in some crops. For example, 000012, 000016, 000018, 000021.
There is also a colour mismatch sometimes between LR and HR: for example 000022.

it seems that the official dataset is unsatisfactory. Therefore, we manually washed x2/x3/x4 datasets to obtain three subsets. There are about 300 damaged image pairs in each original dataset. The washed datasets are now publicly available:

Original Dataset | Original number of images | Ours | Clean Image ID Download Link
------------ | ------------- | ------------- |  ------------
[x2](https://1drv.ms/u/s!AtE0puUOX2nNgW9bGUzZtoRksxwP?e=dP5yMD) | 19000 | 18475 | [Link](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/washed_dataset/x2_clean_img_id.txt)
[x3](https://1drv.ms/u/s!AtE0puUOX2nNgXA8gJT6YRTBhY7x?e=edRCqz) | 19000 | 18643 | [Link](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/washed_dataset/x3_clean_img_id.txt)
[x4](https://1drv.ms/u/s!AtE0puUOX2nNgXHMsFe8G9MbbmQR?e=khQcNg) | 19000 | 18652 | [Link](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/washed_dataset/x4_clean_img_id.txt)

### 4.2 Washed x2+x3 Dataset (AIM_washed_Large) :
Though AIM2020 x2 dataset contains 19K real LR/HR pairs, our models still suffer from overfishing. In light of this, we use x3 LR/HR pairs to fine-tune x2 models. Specifically, we downsample x3 HR images to x2 size (i.e., `HR_img.resize(H//3*2, W//3*2)`), which generates a larger AIM x2 dataset with 37118 images, namely `AIM_washed_Large`. 

This setting contributes to better visualization results on hard samples. Left subfigure is only trained on x2 washed and right subfigure is trained on x2+x3. However, this training strategy results in a chromatism problem. 

<img src="./img/cmp.jpg" width="700" height="315" />

### 4.3 ClipL1 Loss :
To solve the noisy data problem, we propose a new loss function for CNN-based low-level computer vision tasks. As the name implies, ClipL1 Loss combines Clip function and L1 loss. `self.clip_min` sets the gradients of well-trained pixels to zeros and `clip_max` works as a noise filter. 
```python
import torch
import torch.nn as nn

class ClipL1(nn.Module):
    # data range [0, 255], for [0,1] please set clip_min to 1/255=0.003921.
    def __init__(self, clip_min=1.0, clip_max=10.0):
        super(ClipL1, self).__init__()
        self.clip_max = clip_max
        self.clip_min = clip_min

    def forward(self, sr, hr):
        loss = torch.mean(torch.clamp(torch.abs(sr-hr), self.clip_min, self.clip_max))
        return loss
```

## 5. Model Ensemble :
To alleviate the chromatism problem, we use self-ensemble and model ensemble at inference time. Left subfigure is ensembled and right subfigure is a single model baseline.

<img src="./img/cmp2.jpg" width="700" height="315" />

### 5.1 Crop-ensemble

We further propose a new ensemble method called `crop-ensemble`. The motivation is to hide the seam artifact caused by cropping input images:

<img src="./img/cmp3.jpg" width="350" height="450" />

Please refer to `model/__init__.py` Line59 for more information. Different colors of boxes indicate different crop sizes. Small boxes cover the seams between predicted large image patches and vice versa. In our experiments, **crop-ensemble noticeably improves the performance and the more the better!**   

<img src="./img/shave-ensemble.jpg" width="500" height="360" />

## 6. Acknowledgement :
We would like to thank [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch), [DDet](https://github.com/ykshi/DDet), [Pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim), [CBAM](https://github.com/Jongchan/attention-module), [CGD](https://github.com/HolmesShuan/Compact-Global-Descriptor) and [RealSR](https://github.com/Alan-xw/RealSR) for sharing their codes. Our methods are built on those inspiring works. We still borrow some ideas from [NTIRE2019](https://openaccess.thecvf.com/CVPR2019_workshops/CVPR2019_NTIRE) leading methods, such as [OANet](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Du_Orientation-Aware_Deep_Neural_Network_for_Real_Image_Super-Resolution_CVPRW_2019_paper.pdf) and [KPN](https://github.com/csjcai/RealSR). We appreciate the tremendous efforts of previous methods. 

## 7. Cite :
If you find this repository useful, please cite:
```
@misc{AIM2020RealSR,
  author = {Xiangyu He},
  title = {AIM2020-RealSR},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HolmesShuan/AIM2020-RealSR}},
}
```
