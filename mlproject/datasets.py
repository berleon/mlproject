import numpy as np
from PIL import Image


class ClutteredMNIST:
    def __init__(self, dataset, shape=(100, 100), n_clutters=6, clutter_size=8):
        self.dataset = dataset
        self.shape = shape
        self.n_clutters = n_clutters
        self.clutter_size = clutter_size

    def __getitem__(self, idx):
        clutter_pos = []

        def place_clutter():
            rand_idx = np.random.randint(0, len(self.dataset))
            clutter_img = np.array(self.dataset[rand_idx][0])
            h, w = clutter_img.shape

            cs = self.clutter_size
            # select patch
            rh = np.random.randint(0, h - cs)
            rw = np.random.randint(0, w - cs)
            patch = clutter_img[rh:rh+cs, rw:rw+cs]

            # place patch
            rh = np.random.randint(0, self.shape[0] - cs)
            rw = np.random.randint(0, self.shape[1] - cs)
            canvas[rh:rh+cs, rw:rw+cs] = patch
            clutter_pos.append([rh, rw])

        canvas = np.zeros(self.shape, dtype=np.uint8)
        for _ in range(self.n_clutters):
            place_clutter()

        img, label = self.dataset[idx]
        img = np.array(img)
        h, w = img.shape

        rh = np.random.randint(0, self.shape[0] - h)
        rw = np.random.randint(0, self.shape[1] - w)

        canvas[rh:rh+h, rw:rw+w] = img

        return Image.fromarray(canvas, mode='L'), label
