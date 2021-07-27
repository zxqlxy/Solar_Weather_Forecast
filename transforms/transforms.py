import numpy as np


class CenterCrop(object):
    """Crop tthe center of a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        N, h, w = sample.shape
        new_h, new_w = self.output_size

        diff_h = (h - new_h) // 2
        diff_w = (w - new_w) // 2

        return sample[:, diff_h:new_h+diff_h, diff_w:new_w+diff_w]