import os
from data import srdata
import glob

class AIM(srdata.SRData):
    def __init__(self, args, name='AIM', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(AIM, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        # print(names_hr)
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0]))
        )
        # print(names_lr)

        return names_hr, [names_lr]

    def _set_filesystem(self, dir_data):
        super(AIM, self)._set_filesystem(dir_data)
        self.apath = os.path.join(self.apath,'X{}'.format(self.args.scale[0]))
        if self.train:
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.dir_lr = os.path.join(self.apath, 'LR')
        else:
            self.dir_hr = os.path.join(self.apath, 'valid/HR')
            self.dir_lr = os.path.join(self.apath, 'valid/LR')
        if self.input_large: self.dir_lr += 'L'
        # print(self.dir_hr)