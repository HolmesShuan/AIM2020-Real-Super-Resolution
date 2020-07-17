from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch

def make_model(args, parent=False):
    return DWAN(args)

class DWAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DWAN, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        self.scale = scale
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size), act,
                    conv(n_feats, n_feats, kernel_size), act,
                        conv(n_feats, n_feats, kernel_size), act, 
                            conv(n_feats, args.n_colors, kernel_size)]

        # define body module
        self.body_down = nn.Sequential(common.Shuffle_d(scale=scale),
                                    conv(args.n_colors*scale*scale, n_feats, kernel_size))

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        body_tail = [
            nn.PixelShuffle(scale),
            conv(int(n_feats/scale/scale), n_feats, kernel_size),
            # common.CBAM(n_feats),
            nn.ReLU(False),
            conv(n_feats, n_feats, kernel_size),
            # common.CBAM(n_feats),
            nn.ReLU(False)
        ]

        m_tail = [
            conv(args.n_colors*3, n_feats, kernel_size),
            # common.CBAM(n_feats),
            nn.ReLU(False),
            conv(n_feats, n_feats, kernel_size),
            # common.CBAM(n_feats),
            nn.ReLU(False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.body_tail = nn.Sequential(*body_tail)

        self.tail = nn.Sequential(*m_tail)

        # self.dynamic_kernel0 = conv(n_feats, args.n_colors, 7) # common.pixelConv(n_feats, 32, 25, 5, 3)
        # self.dynamic_kernel1 = conv(n_feats, args.n_colors, 5)
        # self.dynamic_kernel2 = conv(n_feats, args.n_colors, 3)

        self.dynamic_kernel0 = common.pixelConv(n_feats, 32, 25, 5, 3)
        self.dynamic_kernel1 = common.pixelConv(n_feats, 49, 49, 7, 3)
        self.dynamic_kernel2 = common.pixelConv(n_feats, 0, 9, 3, 3)


    def forward(self, x):
        with torch.no_grad():
            x = self.sub_mean(x)
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        head_path = self.head(x) + x

        body_down = self.body_down(x)
        body_path = self.body(body_down)
        body_tail = self.body_tail(body_path)

        # dynamic_fmap0 = self.dynamic_kernel0(body_tail) * head_path # self.dynamic_kernel0(body_tail, head_path)
        # dynamic_fmap1 = self.dynamic_kernel1(body_tail) * head_path
        # dynamic_fmap2 = self.dynamic_kernel2(body_tail) * head_path

        dynamic_fmap0 = self.dynamic_kernel0(body_tail, head_path)
        dynamic_fmap1 = self.dynamic_kernel1(body_tail, head_path)
        dynamic_fmap2 = self.dynamic_kernel2(body_tail, head_path)

        dynamic_fmap = torch.cat((dynamic_fmap0, dynamic_fmap1, dynamic_fmap2), 1)

        dynamic_fmap = self.tail(dynamic_fmap)
        dynamic_fmap = self.add_mean(dynamic_fmap)

        return dynamic_fmap 

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

