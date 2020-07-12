# AIM2020-RealSR
Our solution to AIM2020 Real Image Super-Resolution Challenge (x2/x3).

## 1. Basic Models :
Our solution consists of three basic models (**model ensemble**): OADDetv1, OADDetv2 and Deep-OADDet. OADDetv1 and v2 shares the same architecture yet trained with different datasets (further details in [Training Scripts](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/README.md#33-training-scripts) and [Dataset](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/README.md#dataset)). 

<img src="./img/OADDet.jpg" width="500" height="250" />

Our core modules are heavily borrowed from [Inception](https://arxiv.org/pdf/1409.4842.pdf) and [OANet](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Du_Orientation-Aware_Deep_Neural_Network_for_Real_Image_Super-Resolution_CVPRW_2019_paper.pdf) with minor improvements, such as less attention modules, skip connections and LeakyReLU.


## 2. Environment :
We conduct all experiments on Nvidia GPUs (NVIDIA Tesla V100 SXM2 16GB) including training (12 GPUs) and testing (4 GPUs). The total training time is about 1800 GPU hours on V100. It takes about 30GB DRAM during training. The detailed requirements are as follow:
```C
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
```
CUDA_VISIBLE_DEVICES=0 python main.py --model WDDet --n_resblocks 40 --n_feats 128 --res_scale 1.0 --data_test AIM --scale 2 --save AIM_WDDet_x2_VAL_model_latest --test_only --dir_data /nfsdata1/home/hexiangyu/RealSR_X2_Full_Valid_New/ --pre_train /nfsdata1/home/hexiangyu/EDSR-PyTorch-legacy-1.1.0/experiment/AIM_WDDet_x2_Large_Dataset_SSIM_Finetune/model/model_1.pt --n_GPUs 1 --chop --chop-size 410 --shave-size 10
```
### 3.2 Test on your own images:

### 3.3 Training Scripts:
We release all our training scripts to help reproduce our results and hopefully the following methods may benefit from our works.
#### 3.3.1 OADDetv1
```

```
#### 3.3.2 OADDetv2
```
```
#### 3.3.3 Deep-OADDet
```
```

## 4. Dataset :
According to this [issue](https://competitions.codalab.org/forums/21376/3953/):
> I found that many photos in the training dataset are not pixel-wise aligned. Actually, there are different types of misalignment: camera shift, moving objects (e.x. trees, grass).

> However, looking at the dataset, I found that there are very large shifts in some crops. For example, 000012, 000016, 000018, 000021.
There is also a color mismatch sometimes between LR and HR: for example 000022.

it seems that the official dataset is unsatisfactory. Therefore, we manually washed x2/x3/x4 datasets to obtain three subsets. There are about 300 damaged image pairs in each original dataset. The washed datasets are now public available:

Dataset | Original number of images | Ours | Clean Image ID Download Link
------------ | ------------- | ------------- |  ------------
[x2](https://1drv.ms/u/s!AtE0puUOX2nNgW9bGUzZtoRksxwP?e=dP5yMD) | 19000 | 18475 | [Link](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/washed_dataset/x2_clean_img_id.txt)
x3 | 19000 | 18643 | [Link](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/washed_dataset/x3_clean_img_id.txt)
x4 | 19000 | 18652 | [Link](https://github.com/HolmesShuan/AIM2020-RealSR/blob/master/washed_dataset/x4_clean_img_id.txt)

### 4.1 ClipL1 Loss :
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

## 5. Acknowledgement :
We would like to thank [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch), [DDet](https://github.com/ykshi/DDet), [Pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim), [CBAM](https://github.com/Jongchan/attention-module), [CGD](https://github.com/HolmesShuan/Compact-Global-Descriptor) and [RealSR](https://github.com/Alan-xw/RealSR) for sharing their codes. Our methods are built on those inspiring works. We still borrow some ideas from [NTIRE2019](https://openaccess.thecvf.com/CVPR2019_workshops/CVPR2019_NTIRE_search) leading methods, such as [OANet](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Du_Orientation-Aware_Deep_Neural_Network_for_Real_Image_Super-Resolution_CVPRW_2019_paper.pdf) and [KPN](https://github.com/csjcai/RealSR). We appreciate the tremendous efforts of previous methods. 

## 6. Cite :
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
