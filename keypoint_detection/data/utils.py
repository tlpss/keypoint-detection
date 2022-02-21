import abc
import random
import time

import numpy as np
from skimage import io
from torch.utils.data import Dataset


class ImageLoader:
    def get_image(self, path: str, idx: int) -> np.ndarray:
        """
        read the image from disk and return as np array
        """
        # load images @runtime from disk
        image = io.imread(path)
        return image


class BaseImageLoaderDecorator(ImageLoader):
    def __init__(self, image_loader: ImageLoader) -> None:
        self.image_loader = image_loader

    @abc.abstractmethod
    def get_image(self, path: str, idx: int) -> np.ndarray:
        pass


class IOSafeImageLoaderDecorator(BaseImageLoaderDecorator):
    """
    IO safe loader that re-attempts to load image from disk (important for GPULab infrastructure @ UGent)
    """

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__(image_loader)
        self.n_io_attempts = 4

    def get_image(self, path: str, idx: int) -> np.ndarray:
        sleep_time_in_seconds = 1
        for j in range(self.n_io_attempts):
            try:
                image = self.image_loader.get_image(path, idx)
                return image
            except IOError:
                if j == self.n_io_attempts - 1:
                    raise IOError(f"Could not load image for dataset entry with path {path}, index {idx}")

                sleep_time = max(random.gauss(sleep_time_in_seconds, j), 0)
                print(f"caught IOError in {j}th attempt to load image for {path}, sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
                sleep_time_in_seconds *= 2


class CachedImageLoaderDecorator(BaseImageLoaderDecorator):
    """
    Image dataloader that caches the images after the first fetch in np.uint8 format.
    Requires enough CPU Memory to fit entire dataset (img_size^2*3*N_images B)

    This is done lazy instead of prefetching, as the torch dataloader is highly optimized to prefetch data during forward passes etc.
     Impact is expected to be not too big.. TODO -> benchmark.

    Furthermore, this caching requires to set num_workers to 0, as the dataset object is copied by each dataloader worker.
    """

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__(image_loader)

        self.cache = []
        self.cache_index_mapping = {}

    def get_image(self, path: str, idx: int) -> np.ndarray:
        if not path in self.cache_index_mapping:
            img = super().get_image(path, idx)
            self.cache.append(img)
            self.cache_index_mapping.update({path: len(self.cache) - 1})
            return img

        else:
            return self.cache[self.cache_index_mapping[path]]


class ImageDataset(Dataset, abc.ABC):
    def __init__(self, imageloader: ImageLoader = None):
        if imageloader is None:
            self.image_loader = IOSafeImageLoaderDecorator(ImageLoader())

        else:
            self.image_loader = imageloader

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
