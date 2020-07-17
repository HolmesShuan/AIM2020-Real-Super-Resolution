import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.autograd import Variable, Function
from PIL import Image

class CGD(nn.Module):
    def __init__(self, in_channels, bias=True, nonlinear=True):
        super(CGD, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        self.w0 = nn.Parameter(torch.ones(in_channels,1), requires_grad=True)
        self.w1 = nn.Parameter(torch.ones(in_channels,1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(in_channels,1), requires_grad=True)
        
        self.bias0 = nn.Parameter(torch.zeros(1,in_channels,1,1), requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(1,in_channels,1,1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1,in_channels,1,1), requires_grad=True)

        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        # self.tanh = nn.Tanh()

    def forward(self, x):
        b, c, _, _ = x.size()
        x0 = self.avg_pool(x).view(b, c, 1, 1)
        x1 = self.max_pool(x).view(b, c, 1, 1)
        
        x0_s = self.softmax(x0) # b ,c ,1 ,1 

        y0 = torch.matmul(x0.view(b,c), self.w0).view(b, 1, 1, 1)
        y1 = torch.matmul(x1.view(b,c), self.w1).view(b, 1, 1, 1)

        y0_s = torch.tanh(y0*x0_s + self.bias0) # b ,c ,1 ,1 
        y1_s = torch.tanh(y1*x0_s + self.bias1) # b ,c ,1 ,1 

        y2 = torch.matmul(y1_s.view(b,c), self.w2).view(b, 1, 1, 1)
        y2_s = torch.tanh(y2*y0_s + self.bias2).view(b, c, 1, 1)

        z = x*(y2_s+1)
 
        return z

class bicubic(nn.Module):
    def __init__(self):
        super(bicubic,self).__init__()
    def cubic(self,x):
        absx = torch.abs(x)
        absx2 = torch.abs(x)*torch.abs(x)
        absx3 = torch.abs(x)*torch.abs(x)*torch.abs(x)

        condition1 = (absx<=1).to(torch.float32)
        condition2 = ((1<absx)&(absx<=2)).to(torch.float32)
        
        f = (1.5*absx3 - 2.5*absx2 +1)*condition1+(-0.5*absx3 + 2.5*absx2 -4*absx +2)*condition2
        return f
    def contribute(self,in_size,out_size,scale):
        kernel_width = 4
        if scale<1:
            kernel_width = 4/scale
        x0 = torch.arange(start = 1,end = out_size[0]+1).to(torch.float32)
        x1 = torch.arange(start = 1,end = out_size[1]+1).to(torch.float32)
        
        u0 = x0/scale + 0.5*(1-1/scale)
        u1 = x1/scale + 0.5*(1-1/scale)

        left0 = torch.floor(u0-kernel_width/2)
        left1 = torch.floor(u1-kernel_width/2)

        P = np.ceil(kernel_width)+2
        
        indice0 = left0.unsqueeze(1) + torch.arange(start = 0,end = P).to(torch.float32).unsqueeze(0)
        indice1 = left1.unsqueeze(1) + torch.arange(start = 0,end = P).to(torch.float32).unsqueeze(0)
        
        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale* self.cubic(mid0*scale)
            weight1 = scale* self.cubic(mid1*scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0/(torch.sum(weight0,2).unsqueeze(2))
        weight1 = weight1/(torch.sum(weight1,2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]),indice0),torch.FloatTensor([in_size[0]])).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]),indice1),torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0,0)[0][0]
        kill1 = torch.eq(weight1,0)[0][0]
        
        weight0 = weight0[:,:,kill0==0]
        weight1 = weight1[:,:,kill1==0]

        indice0 = indice0[:,:,kill0==0]
        indice1 = indice1[:,:,kill1==0]


        return weight0,weight1,indice0,indice1

    def forward(self,input, scale = 1/4):
        [b,c,h,w] = input.shape
        output_size = [b,c,int(h*scale),int(w*scale)]

        weight0,weight1,indice0,indice1 = self.contribute([h,w],[int(h*scale),int(w*scale)],scale)

        weight0 = np.asarray(weight0[0],dtype = np.float32)
        weight0 = torch.from_numpy(weight0).cuda()

        indice0 = np.asarray(indice0[0],dtype = np.float32)
        indice0 = torch.from_numpy(indice0).cuda().long()
        out = input[:,:,(indice0-1),:]*(weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out,dim = 3))
        A = out.permute(0,1,3,2)

        weight1 = np.asarray(weight1[0],dtype = np.float32)
        weight1 = torch.from_numpy(weight1).cuda()

        indice1 = np.asarray(indice1[0],dtype = np.float32)
        indice1 = torch.from_numpy(indice1).cuda().long()
        out = A[:,:,(indice1-1),:]*(weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = torch.round(255*torch.sum(out,dim = 3).permute(0,1,3,2))/255

        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp0 = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.mlp1 = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp0( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp1( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp0( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp1( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg'], no_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), padding_mode='replicate',bias=bias)
'''
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
'''
class MeanShift(nn.Module):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__()
        self.register_buffer('rgb_mean', torch.zeros(3))
        # self.register_buffer('rgb_std', torch.ones(3))
        self.rgb_mean =  rgb_range * torch.Tensor(rgb_mean) 
        # self.rgb_std = torch.Tensor(rgb_std)
        if sign < 0:
            self.rgb_mean = -self.rgb_mean # /self.rgb_std
            # self.rgb_std = 1/self.rgb_std
    def forward(self,x):
        mean = x.new(self.rgb_mean).view(1,3,1,1).expand_as(x)
        # std = x.new(self.rgb_std).view(1,3,1,1).expand_as(x)
        x = x.add(mean)
        return x

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        if self.res_scale == 1.0:
            res = self.body(x)
        else:
            res = self.body(x).mul(self.res_scale)
        res += x

        return res

class OAModule(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True, use_att=True):

        super(OAModule, self).__init__()
        
        self.dia_conv = nn.Conv2d(n_feats, 96, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=bias)
        self.hor_conv = nn.Conv2d(n_feats, 64, kernel_size=(1,5), stride=(1,1), padding=(0,2), dilation=1, groups=1, bias=bias)
        self.ver_conv = nn.Conv2d(n_feats, 64, kernel_size=(5,1), stride=(1,1), padding=(2,0), dilation=1, groups=1, bias=bias)
        if use_att:
            self.conv = nn.Sequential(CBAM(224), nn.ReLU(False), 
                                nn.Conv2d(224, n_feats, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=bias))
        else:
            self.conv = nn.Sequential(nn.ReLU(True), 
                                nn.Conv2d(224, n_feats, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=bias))

    def forward(self, x):
        res = x
        dia_conv = self.dia_conv(x)
        hor_conv = self.hor_conv(x)
        ver_conv = self.ver_conv(x)
        conv = torch.cat((dia_conv, hor_conv, ver_conv), 1)
        conv_tail = self.conv(conv)
        return res + conv_tail

class WOAModule(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True, use_wider_kernel_size=True):
 
        super(WOAModule, self).__init__()
        
        if use_wider_kernel_size:
            self.dia_conv = nn.Conv2d(n_feats, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=bias)
            self.hor_conv = nn.Conv2d(n_feats, 128, kernel_size=(1,5), stride=(1,1), padding=(0,2), dilation=1, groups=1, bias=bias)
            self.ver_conv = nn.Conv2d(n_feats, 128, kernel_size=(5,1), stride=(1,1), padding=(2,0), dilation=1, groups=1, bias=bias)
        else:
            self.dia_conv = nn.Conv2d(n_feats, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=bias)
            self.hor_conv = nn.Conv2d(n_feats, 128, kernel_size=(1,3), stride=(1,1), padding=(0,1), dilation=1, groups=1, bias=bias)
            self.ver_conv = nn.Conv2d(n_feats, 128, kernel_size=(3,1), stride=(1,1), padding=(1,0), dilation=1, groups=1, bias=bias)
        self.conv = nn.Sequential(CBAM(128*3, pool_types=['avg', 'max']),
                                    nn.ReLU(True), 
                                    nn.Conv2d(128*3, n_feats, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=bias))

    def forward(self, x):
        dia_conv = self.dia_conv(x)
        hor_conv = self.hor_conv(x)
        ver_conv = self.ver_conv(x)
        conv = torch.cat((dia_conv, hor_conv, ver_conv), 1)
        conv_tail = self.conv(conv)
        return conv_tail

class FExtractModulev2(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True, use_wider_kernel_size=True, use_att=False):
        super(FExtractModulev2, self).__init__()
        
        if use_wider_kernel_size:
            self.dia_conv = nn.Conv2d(n_feats, 48, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=bias)
            self.hor_conv = nn.Conv2d(n_feats, 96, kernel_size=(1,5), stride=(1,1), padding=(0,2), dilation=1, groups=1, bias=bias)
            self.ver_conv = nn.Conv2d(n_feats, 48, kernel_size=(5,1), stride=(1,1), padding=(2,0), dilation=1, groups=1, bias=bias)
        else:
            self.dia_conv = nn.Conv2d(n_feats, 48, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=bias)
            self.hor_conv = nn.Conv2d(n_feats, 96, kernel_size=(1,3), stride=(1,1), padding=(0,1), dilation=1, groups=1, bias=bias)
            self.ver_conv = nn.Conv2d(n_feats, 48, kernel_size=(3,1), stride=(1,1), padding=(1,0), dilation=1, groups=1, bias=bias)
        if use_att:
            self.conv = nn.Sequential(CBAM(96*2), nn.ReLU(False), 
                                nn.Conv2d(96*2, n_feats, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=bias))
        else:
            self.conv = nn.Sequential(nn.ReLU(True), 
                                nn.Conv2d(96*2, n_feats, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=bias))

    def forward(self, x):
        dia_conv = self.dia_conv(x)
        hor_conv = self.hor_conv(x)
        ver_conv = self.ver_conv(x)
        conv = torch.cat((dia_conv, hor_conv, ver_conv), 1)
        conv_tail = self.conv(conv) + x
        return conv_tail

class FExtractModule(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True, use_wider_kernel_size=True):
        super(FExtractModule, self).__init__()
        
        if use_wider_kernel_size:
            self.dia_conv = nn.Conv2d(n_feats, 48, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=bias)
            self.hor_conv = nn.Conv2d(n_feats, 96, kernel_size=(1,5), stride=(1,1), padding=(0,2), dilation=1, groups=1, bias=bias)
            self.ver_conv = nn.Conv2d(n_feats, 48, kernel_size=(5,1), stride=(1,1), padding=(2,0), dilation=1, groups=1, bias=bias)
        else:
            self.dia_conv = nn.Conv2d(n_feats, 48, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=bias)
            self.hor_conv = nn.Conv2d(n_feats, 96, kernel_size=(1,3), stride=(1,1), padding=(0,1), dilation=1, groups=1, bias=bias)
            self.ver_conv = nn.Conv2d(n_feats, 48, kernel_size=(3,1), stride=(1,1), padding=(1,0), dilation=1, groups=1, bias=bias)
        self.conv = nn.Sequential(nn.ReLU(False), nn.Conv2d(96*2, n_feats, kernel_size=kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=bias))

    def forward(self, x):
        dia_conv = self.dia_conv(x)
        hor_conv = self.hor_conv(x)
        ver_conv = self.ver_conv(x)
        conv = torch.cat((dia_conv, hor_conv, ver_conv), 1)
        conv_tail = self.conv(conv)
        return conv_tail

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)

class pixelConv(nn.Module):
    # Generate pixel kernel  (3*k*k)xHxW
    def __init__(self, in_feats, mid_feats, out_feats=3, ksize=3, fmap_channels=3):
        super(pixelConv,self).__init__()
        self.padding = (ksize-1)//2
        self.ksize = ksize
        self.zero_padding = nn.ZeroPad2d(self.padding)
        self.fmap_channels = fmap_channels

        assert out_feats == ksize**2, "wrong output channel number"
        
        if mid_feats != 0:
            self.kernel_conv =nn.Sequential(*[
                nn.Conv2d(in_feats, mid_feats, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(mid_feats, fmap_channels*out_feats, kernel_size=3, padding=1)
            ])
        else:
            self.kernel_conv =nn.Sequential(*[
                nn.Conv2d(in_feats, fmap_channels*out_feats, kernel_size=3, padding=1)
            ])
  
    def forward(self, x_feature, x):
        kernel_set = self.kernel_conv(x_feature)
        dtype = kernel_set.data.type()
        ks = self.ksize
        N = self.ksize**2 # patch size 
        # padding the input image with zero values
        if self.padding:
            x = self.zero_padding(x)
            
        p = self._get_index(kernel_set,dtype)
        p = p.contiguous().permute(0, 2, 3, 1).long()
        x_pixel_set = self._get_x_q(x, p, N)
        b,c,h,w = kernel_set.size()
        kernel_set_reshape = kernel_set.reshape(-1,self.ksize**2,self.fmap_channels,h,w).permute(0,2,3,4,1)
        x_ = x_pixel_set
     
        out = x_*kernel_set_reshape
        out = out.sum(dim=-1,keepdim=True).squeeze(dim=-1)
        return out 

    def _get_index(self, kernel_set, dtype):
        '''
        get absolute index of each pixel in image
        '''
        N, b, h, w = self.ksize**2, kernel_set.size(0), kernel_set.size(2), kernel_set.size(3)
        # get absolute index of center index
        p_0_x, p_0_y = np.meshgrid(range(self.padding, h + self.padding), range(self.padding, w + self.padding), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1) 
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        # get relative index around center pixel
        p_n_x, p_n_y = np.meshgrid(range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1),
                                   range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False) 
        p = p_0 + p_n
        p = p.repeat(b,1,1,1)
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()  # dimension of q: (b,h,w,2N)
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*padded_w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        # index_x*w + index_y
        index = q[..., :N] * padded_w + q[...,N:] 

        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset