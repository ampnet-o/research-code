import numpy as np

class Bitmap:
    def __init__(self, value: np.ndarray) -> None:
        assert value.dtype == np.bool
        assert len(value.shape) == 1

        self.N_dat = value.shape[0]
        self.dat = np.packbits(value)

        assert self.dat.dtype == np.uint8

    def __getitem__(self, key: int | np.ndarray):
        if type(key) == int:
            key = np.array([key])

        outer = key // 8
