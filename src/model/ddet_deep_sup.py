from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch

def make_model(args, parent=False):
    return DDet_SUP(args)

class DDet_SUP(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DDet_SUP, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        self.scale = scale
        act = nn.LeakyReLU(0.2, True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module

        # m_head = [conv(args.n_colors, n_feats, 5), act,
        #             conv(n_feats, n_feats, kernel_size), act,
        #                 conv(n_feats, n_feats, kernel_size), act, 
        #                     conv(n_feats, n_feats, kernel_size), act,
        #                         conv(n_feats, n_feats, kernel_size), act, 
        #                             conv(n_feats, n_feats, kernel_size), act,
        #                                 conv(n_feats, n_feats, kernel_size), act, 
        #                                     conv(n_feats, args.n_colors, 1)]


        m_head = [conv(args.n_colors, n_feats, 5), act,
                    common.FExtractModule(n_feats, kernel_size, bias=True, use_wider_kernel_size=True), act,
                        common.FExtractModule(n_feats, kernel_size, bias=True, use_wider_kernel_size=True), act, 
                            common.FExtractModule(n_feats, kernel_size, bias=True, use_wider_kernel_size=True), 
                                common.CBAM(n_feats), act, 
                                    conv(n_feats, args.n_colors, 1)]

        # define body module
        self.body_down = nn.Sequential(common.Shuffle_d(scale=scale),
                                    conv(args.n_colors*scale*scale, n_feats, kernel_size))

        m_body = [
            common.OAModule(
                n_feats, kernel_size, use_att=i>=n_resblocks//4*3 ) for i in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        body_tail = [
            nn.PixelShuffle(scale),
            conv(int(n_feats//scale//scale), n_feats, kernel_size),
            nn.ReLU(True),
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        ]

        m_tail = [
            conv(args.n_colors*3, n_feats, kernel_size),
            # common.CBAM(n_feats),
            nn.ReLU(True),
            conv(n_feats, args.n_colors, kernel_size),
            # common.CBAM(n_feats),
            # nn.ReLU(False),
            # conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.body_tail = nn.Sequential(*body_tail)

        self.tail = nn.Sequential(*m_tail)

        # self.dynamic_kernel0 = conv(n_feats, 64, 7) # common.pixelConv(n_feats, 32, 25, 5, 3)
        # self.dynamic_kernel1 = conv(n_feats, 64, 5)
        # self.dynamic_kernel2 = conv(n_feats, 64, 3)

        self.dynamic_kernel0 = common.pixelConv(n_feats, 64, 25, 5, 3)
        self.dynamic_kernel1 = common.pixelConv(n_feats, 64, 49, 7, 3)
        self.dynamic_kernel2 = common.pixelConv(n_feats, 64, 9, 3, 3)

        self.upsampler = common.bicubic()


    def forward(self, x):
        with torch.no_grad():
            x = self.upsampler(x, self.scale)
            x = self.sub_mean(x)
            # x = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        head_path = self.head(x) + x
        output = self.add_mean(head_path)

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

        return output, dynamic_fmap 

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

