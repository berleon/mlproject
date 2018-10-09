import numpy as np
from PIL import Image


class ClutteredMNIST:
    def __init__(self, dataset, shape=(100, 100), n_clutters=6, clutter_size=8, n_samples=60000):
        self.dataset = dataset
        self.shape = shape
        self.n_clutters = n_clutters
        self.clutter_size = clutter_size
        self.n_samples = n_samples
        self.parameters = self.generate_parameters()

    def generate_parameters(self):
        all_params = []
        h, w = self.dataset[0][0].shape
        for i in range(self.n_samples):
            params = {
                'idx': i % len(self.dataset),
                'digit_h': np.random.randint(0, self.shape[0] - h),
                'digit_w': np.random.randint(0, self.shape[1] - w),
            }
            clutter = []
            for _ in self.n_clutters:
                clutter_idx = np.random.randint(0, len(self.dataset))
                cs = self.clutter_size
                ph = np.random.randint(0, h - cs)
                pw = np.random.randint(0, w - cs)
                ch = np.random.randint(0, self.shape[0] - cs)
                cw = np.random.randint(0, self.shape[1] - cs)
                clutter.append({
                    'clutter_idx': clutter_idx,
                    'patch_h': ph,
                    'patch_w': pw,
                    'clutter_h': ch,
                    'clutter_w': cw,
                })
            params['clutter'] = clutter
            all_params.append(params)
        return all_params

    def __getitem__(self, idx):
        canvas = np.zeros(self.shape, dtype=np.uint8)
        params = self.parameters[idx]
        for clutter in params['clutter']:
            clutter_img = np.array(self.dataset[clutter['clutter_idx']][0])
            h, w = clutter_img.shape
            # select patch
            cs = self.clutter_size
            ph = clutter['patch_h']
            pw = clutter['patch_w']
            patch = clutter_img[ph:ph+cs, pw:pw+cs]
            # place patch
            ch = clutter['clutter_h']
            cw = clutter['clutter_w']
            canvas[ch:ch+cs, cw:cw+cs] = patch

        img, label = self.dataset[params['idx']]
        img = np.array(img)
        h, w = img.shape
        dh = params['digit_h']
        dw = params['digit_w']
        canvas[dh:dh+h, dw:dw+w] = img
        return Image.fromarray(canvas, mode='L'), label
