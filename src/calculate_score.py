import os
import shutil
from PIL import Image
import PIL
import numpy as np
import cv2
import torch
import numpy as np
import torch.nn.functional as F

def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, 
          data_range, 
          win, 
          size_average=True, 
          K=(0.01,0.03)):
          
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten( cs_map, 2 ).mean(-1)
    return ssim_per_channel, cs


def ssim(X, Y, 
         data_range=255, 
         size_average=True, 
         win_size=11, 
         win_sigma=1.5, 
         win=None, 
         K=(0.01, 0.03), 
         nonnegative_ssim=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same shape.')
    
    if win is not None: # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')
    
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    
    ssim_per_channel, cs = _ssim(X, Y,
                                data_range=data_range,
                                win=win,
                                size_average=False,
                                K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)
    
    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)

def calc_psnr(sr, hr, scale=2, rgb_range=255, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def calc_ssim(sr, hr, scale, rgb_range):
    if hr.nelement() == 1: return 0

    shave = scale + 6

    X = sr[..., shave:-shave, shave:-shave]
    Y = hr[..., shave:-shave, shave:-shave]

    return ssim(X, Y, 
         data_range=rgb_range, 
         size_average=True, 
         win_size=11, 
         win_sigma=1.5, 
         win=None, 
         K=(0.01, 0.03), 
         nonnegative_ssim=False)


predicted_img_dir = '/nfsdata1/AIM/X2/valid/HR'
ground_truth_dir = ''

idx = 0
score = 0.
for root, dirs, files in os.walk(predicted_img_dir):
    for name in files:
        if name[-3:] == 'png':
            print('Processing : ', name)
            idx += 1
            in_sr_path = os.path.join(root, name)
            sr = Image.open(in_sr_path)
            sr = np.array(sr, dtype=np.float32)
            sr = torch.from_numpy(sr).float().cuda()

            in_hr_path = os.path.join(ground_truth_dir, in_sr_path.split('/')[-1].replace('.', '_HR.'))
            hr = Image.open(in_hr_path)
            hr = np.array(hr, dtype=np.float32)
            hr = torch.from_numpy(hr).float().cuda()

            score += 0.5*calc_psnr(sr, hr, scale=2, rgb_range=255)/50. + 0.5*(calc_ssim(sr, hr, scale=2, rgb_range=255)-0.4)/0.6
            
print('Score :', score/idx)