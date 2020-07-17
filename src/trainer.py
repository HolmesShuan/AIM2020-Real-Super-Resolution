import os
import math
from decimal import Decimal

import utility
import numpy as np
import numpy

import torch
import torch.nn.utils as utils
from tqdm import tqdm
from cv2.ximgproc import guidedFilter
import cv2

import torch.backends.cudnn as cudnn

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        if args.deep_supervision:
            self.deep_supervision = True
            self.deep_supervision_factor = args.deep_supervision_factor
        else:
            self.deep_supervision = False
            self.deep_supervision_factor = 0.

        cudnn.benchmark = True
        cudnn.deterministic = False

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            if self.deep_supervision:
                sr_im, sr = self.model(lr, 0)
                loss = self.loss(sr, hr) + self.deep_supervision_factor*self.loss(sr_im, hr)
            else:
                sr = self.model(lr, 0)
                loss = self.loss(sr, hr) 
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

        if self.args.data_train[0] == 'DIV2K' or self.args.test_epoch == 0:
            self.ckp.save(self, epoch, is_best=True)

    def test(self):
        epoch = self.optimizer.get_last_epoch()

        torch.set_grad_enabled(False)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    if self.args.guided_filtering:
                        sr_cpu = sr.squeeze().permute(1,2,0).detach().cpu().numpy()
                        sr_cpu = sr_cpu.astype(np.uint8)
                        if self.args.guided_type == 'RGB':
                            guide = sr_cpu
                        elif self.args.guided_type == 'Gray':
                            guide = cv2.cvtColor(sr_cpu, cv2.COLOR_RGB2GRAY)
                        elif self.args.guided_type == 'Ycbcr':
                            guide = cv2.cvtColor(sr_cpu, cv2.COLOR_BGR2YCR_CB)
                            guide = guide[:,:,0]
                        elif self.args.guided_type == 'HSV':
                            guide = cv2.cvtColor(sr_cpu, cv2.COLOR_RGB2HSV)
                            # sr_cpu = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                        else:
                            guide = sr_cpu
                        guided_img = guidedFilter(guide, sr_cpu, self.args.guided_radius, self.args.guided_eps)
                        sr = torch.from_numpy(guided_img).permute(2,0,1).unsqueeze(0).float().to(sr.device)
                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += 0.5*utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )/50. + 0.5*(utility.calc_ssim(sr, hr, scale, self.args.rgb_range)-0.4)/0.6
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tScore: {:.6f} (Best: {:.6f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

