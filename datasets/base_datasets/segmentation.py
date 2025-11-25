# coding=utf-8

import os
from typing import Callable, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from tools.utils import find_files

__all__ = ["SegmentationDataset"]


class SegmentationDataset(Dataset):
    """
    Basic dataset for semantic segmentation. An example of file structure is as followings.

    .. data tree

        ├── ....
        │   ├── dataset_name
        │   │   ├── image
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── label
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    Params:
        root_path: str. The root path of the dataset folder.
        transforms: callable. The augmentors for images and labels.
        test_mode: bool. Labels will not be loaded if `test_mode` is true. Default: false
        image_dir: str. Image directory of `root_path`. Default: 'images'
        image_suffix str. Suffix of images. Default: '.jpg'
        label_dir: str. Label directory of `root_path`. Default: 'labels'
        label_suffix str. Suffix of labels. Default: '.jpg'
        image_mean: list. The RGB average value of whole dataset. Default (0.485, 0.456, 0.406)
        image_std: list. The RGB standard deviation value of whole dataset. Default (0.229, 0.224, 0.225)
        div_std: bool. Whether the image data will be divided by `std`. Default True
        chw_format: bool. If True, the image shape format is CxHxW, otherwise HxWxC. Default True
    """

    CLASSES = ()
    PALETTE = []

    def __init__(
        self,
        root_path: str,
        transforms: Optional[Callable] = None,
        test_mode=False,
        image_dir="images",
        image_suffix=".jpg",
        label_dir="labels",
        label_suffix=".jpg",
        image_mean=(0.485, 0.456, 0.406),  # for imagenet
        image_std=(0.229, 0.224, 0.225),  # for imagenet
        div_std=True,
        chw_format=True,
    ):
        self.root_path = os.path.expanduser(root_path)
        self.transforms = transforms
        self.test_mode = test_mode
        self.image_dir = image_dir
        self.image_suffix = image_suffix
        self.label_dir = label_dir
        self.label_suffix = label_suffix
        self.image_mean = image_mean
        self.image_std = image_std
        self.div_std = div_std
        self.chw_format = chw_format

        # Load images and labels
        self.image_path = find_files(
            directory=os.path.join(self.root_path, self.image_dir),
            pattern="*" + self.image_suffix,
        )
        self.image_path = sorted(list(self.image_path))

        if not self.test_mode:
            self.label_path = find_files(
                directory=os.path.join(self.root_path, self.label_dir),
                pattern="*" + self.label_suffix,
            )
            self.label_path = sorted(list(self.label_path))
            # assert len(self.image_path) == len(self.label_path)
        else:
            self.label_path = list()

        # get the number of classes
        self.num_classes = len(self.CLASSES)

    def encode_label(self, label):
        """
        Convert the label image into a trainable label according to the index value of RGB of each class.

        Params:
            label: 3-D numpy array. RGB images to be encoded. Note that the ignored category is encoded as 255

        Return:
            return the encoded 2-D label
        """
        result = np.ones(label.shape[0:2], dtype=np.int16) * 255
        for i, cls in enumerate(self.PALETTE):
            result[np.where(np.all(label == self.PALETTE[i], axis=-1))[:2]] = i
        return result

    def decode_label(self, label):
        """
        Convert the predict label into a label image according to the index value of RGB of each class.

        Params:
            label: 2-D numpy array. Label to de decoded. Note that the ignored category will be decoded as 255

        Return:
            return the decoded 3-D label
        """
        result = np.ones((*label.shape[0:2], 3), dtype=np.uint8) * 255
        for i, cls in enumerate(self.PALETTE):
            result[label == i, :] = self.PALETTE[i]
        return result

    def load_image(self, path):
        """load image by opencv"""
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_label(self, path):
        """load label by opencv"""
        label = cv2.imread(path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        return label

    def __getitem__(self, index):
        """Get item by index"""
        if not self.test_mode:
            # Load image and label
            image = self.load_image(self.image_path[index])
            label = self.load_label(self.label_path[index])

            assert len(image.shape) in [2, 3]
            size = image.shape[0:2]

            # Encode label
            label = self.encode_label(label)

            # Data augmentation
            if self.transforms is not None:
                image, label = self.transforms(image, label)

            # Normalization
            image = image / 255.0
            image = image - self.image_mean
            if self.div_std:
                image = image / self.image_std

            # Channel transpose
            if len(image.shape) == 3 and self.chw_format:
                image = image.transpose([2, 0, 1])

            if len(label.shape) == 3 and self.chw_format:
                label = label.transpose([2, 0, 1])

            return {
                "image": np.array(image, dtype=np.float32),
                "label": np.array(label, dtype=np.int64),
                "size": np.array(size, dtype=np.int32),
                "index": np.array(index, dtype=np.int32),
            }
        else:
            # Open image
            image_path = self.image_path[index]
            image = self.load_image(image_path)

            assert len(image.shape) in [2, 3]
            size = image.shape[0:2]

            # Data augmentation
            if self.transforms is not None:
                image, label = self.transforms(image, label)

            # Normalization
            image = image / 255.0
            image = image - self.image_mean
            if self.div_std:
                image = image / self.image_std

            # Channel transpose
            if len(image.shape) == 3 and self.chw_format:
                image = image.transpose([2, 0, 1])

            return {
                "image": np.array(image, dtype=np.float32),
                "size": np.array(size, dtype=np.int32),
                "index": np.array(index, dtype=np.int32),
            }

    def __len__(self):
        return len(self.image_path)
