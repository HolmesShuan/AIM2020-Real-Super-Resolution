from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch

def make_model(args, parent=False):
    return DOANet(args)

class DOANet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DOANet, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        self.scale = scale
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define body module
        self.body_down = nn.Sequential(common.Shuffle_d(scale=scale),
                                    conv(args.n_colors*scale*scale, n_feats, kernel_size))

        m_body = [common.WOAModule( n_feats, kernel_size, use_wider_kernel_size=True ) 
                            for i in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        body_tail = [
            conv(n_feats, 288, kernel_size),
            nn.PixelShuffle(scale),
            conv(int(288//scale//scale), 64, kernel_size),
            nn.ReLU(True),
            conv(64, args.n_colors, kernel_size)
        ]

        self.body = nn.Sequential(*m_body)
        self.body_tail = nn.Sequential(*body_tail)

        self.upsampler = common.bicubic()

    def forward(self, x):
        with torch.no_grad():
            x = self.upsampler(x, self.scale)
            x = self.sub_mean(x)
            
        body_down = self.body_down(x)
        body_path = self.body(body_down) + body_down
        body_tail = self.body_tail(body_path)

        body_tail = self.add_mean(body_tail)
        return body_tail 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

