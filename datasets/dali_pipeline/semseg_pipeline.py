# coding=utf-8

from typing import List, Optional, Tuple

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def

from datasets.base_datasets import SegmentationDataset

__all__ = [
    "SemSegExternalInputCallable",
    "semseg_training_pipeline",
    "semseg_validation_pipeline",
]


class SemSegExternalInputCallable:
    """External input callable object for NVIDIA DALI pipeline

    Before every iteration, DALI External Source operator queries its source parameter
    for new data to pass it further for processing in the pipeline. The time necessary to
    obtain data from source by calling it (when the source is callable) or calling next(source)
    (in case of iterabe) can be significant and it can impact the time to process an iteration
    - especially as it's a blocking operation in the main Python thread.

    Reference: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html
    """

    def __init__(
        self,
        dataset: SegmentationDataset,
        batch_size: int,
        shard_id: int = 0,
        num_shards: int = 1,
        random_shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        sampling_indices: Optional[List] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.random_shuffle = random_shuffle
        self.shuffle_seed = shuffle_seed

        if sampling_indices is None:
            self.sampling_indices = list(range(len(self.dataset)))
        else:
            for idx in sampling_indices:
                assert idx < len(self.dataset), "index error occurred"
            self.sampling_indices = sampling_indices

        self.num_samples = len(self.sampling_indices)
        self.image_paths = [self.dataset.image_path[i] for i in self.sampling_indices]
        self.label_paths = [self.dataset.label_path[i] for i in self.sampling_indices]

        # If the dataset size is not divisibvle by number of shards,
        # the trailing samples will be omitted.
        self.shard_size = self.num_samples // num_shards
        self.shard_offset = self.shard_size * shard_id

        # If the dataset size is not divisible by the batch size, the last
        # incomplete batch will be omitted.
        self.full_iterations = (self.shard_size + batch_size - 1) // batch_size

        self.perm = None  # permutation of indices
        self.last_seen_epoch = (
            # so that we don't have to recompute the `self.perm` for every sample
            None
        )

    def __call__(self, sample_info):
        # Indicate end of the epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()

        # indices permutation per epoch
        if self.random_shuffle and self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=self.shuffle_seed)
            self.perm = self.perm.permutation(self.num_samples)

        # get sample index
        sample_idx = (sample_info.idx_in_epoch + self.shard_offset) % self.num_samples
        if self.random_shuffle:
            assert self.perm is not None, "The perm function is None"
            sample_idx = self.perm[sample_idx]

        # get images and labels
        image_path = self.image_paths[sample_idx]
        label_path = self.label_paths[sample_idx]
        encoded_image = np.fromfile(image_path, dtype=np.uint8)
        encoded_label = np.fromfile(label_path, dtype=np.uint8)
        sample_idx = np.array(sample_idx, dtype=np.int64)
        return encoded_image, encoded_label, sample_idx

    def __len__(self):
        return self.full_iterations


def encode_rgb888(r, g, b):
    """An 8-bit RGB color is encoded into the 16-bit RGB565 format
    and then merged into a 16-bit integer."""
    return r // 8 * 2048 + g // 4 * 32 + b // 8


@pipeline_def
def semseg_training_pipeline(
    dataset: SegmentationDataset,
    batch_size_extsrc: int,
    output_size: Tuple[int, int],
    sampling_indices: Optional[List] = None,
    num_shards: int = 1,
    shard_id: int = 0,
    random_shuffle: bool = False,
    shuffle_seed: int = 0,
    random_scale_prob: float = 1.0,
    random_scale_limit: Tuple[float, float] = (-0.5, 1.0),
    color_jitter_prob: float = 0.0,
    color_brightness: Tuple[float, float] = (0.8, 1.2),
    color_contrast: Tuple[float, float] = (0.8, 1.2),
    color_saturation: Tuple[float, float] = (0.8, 1.2),
    color_hue: Tuple[float, float] = (-0.1, 0.1),
    horizontal_flip_prob: float = 0.5,
):
    """Defines a DALI pipeline for semantic segmentation training with synchronized image-label augmentation.

    Params:
        dataset: SegmentationDataset object containing images and RGB-encoded labels
        batch_size_extsrc: Batch size for external data source (must match pipeline `batch_size`)
        output_size: Target output dimensions (H, W), `None` preserves original size
        sampling_indices: Optional subset indices for partial dataset sampling
        num_shards: Number of shards for distributed training
        shard_id: Current shards index for distributed training
        random_shuffle: Enable shuffling of dataset samples
        shuffle_seed: Seed for reproducible shuffling
        random_scale_prob: Probability to apply random scaling augmentation
        random_scale_limit: Scaling factor range for random reisze
        color_jitter_prob: Probability to apply color jitter
        color_brightness: Brightness adjustment range
        color_contrast: Contrast adjustment range
        color_saturation: Saturation adjustment range
        color_hue: Hue adjustment range
        horizontal_flip_prob: Probability of horizontal flip

    Returns:
        DALI pipeline object containing:
        - Augmented images (float32 tensor)
        - Processed segmentation masks (int32 tensor)
        - Original image shapes (for visualization)
        - Sample indices (for debugging)
    """
    external_source = SemSegExternalInputCallable(
        dataset=dataset,
        num_shards=num_shards,
        batch_size=batch_size_extsrc,
        shard_id=shard_id,
        random_shuffle=random_shuffle,
        shuffle_seed=shuffle_seed,
        sampling_indices=sampling_indices,
    )
    # Note: ``num_outputs`` is not compatible with named ``ExternalSource``
    jpegs, label, index = fn.external_source(
        source=external_source,
        num_outputs=3,
        batch=False,  # must be false
        parallel=True,
        dtype=[types.UINT8, types.UINT8, types.INT64],
    )
    jpegs_shape = fn.peek_image_shape(jpegs)
    image = fn.decoders.image(jpegs, device="mixed", output_type=types.DALIImageType.RGB)
    label = fn.decoders.image(label, device="mixed", output_type=types.DALIImageType.RGB)

    # encode RGB label
    r = fn.cast(label[:, :, 0:1], dtype=types.UINT16)
    g = fn.cast(label[:, :, 1:2], dtype=types.UINT16)
    b = fn.cast(label[:, :, 2:3], dtype=types.UINT16)
    int_label = encode_rgb888(r, g, b)
    lookup_keys = [encode_rgb888(r, g, b) for (r, g, b) in external_source.dataset.PALETTE]
    lookup_values = list(range(len(external_source.dataset.PALETTE)))
    assert len(set(lookup_keys)) == len(lookup_keys), "Duplicate key found."

    label = fn.lookup_table(int_label, default_value=255, keys=lookup_keys, values=lookup_values)

    # random resize and crop
    if random_scale_prob > 0:
        resize_scale = fn.random.uniform(range=[1 + s for s in random_scale_limit], dtype=types.FLOAT)
        apply_scale = fn.random.coin_flip(probability=random_scale_prob)
        apply_scale = fn.cast(apply_scale, dtype=types.FLOAT)
        resize_y = fn.cast(jpegs_shape[0], dtype=types.FLOAT) * resize_scale * apply_scale

        image = fn.resize(image, resize_x=0, resize_y=resize_y, interp_type=types.INTERP_CUBIC)
        label = fn.resize(label, resize_x=0, resize_y=resize_y, interp_type=types.INTERP_NN)

        crop_pos_x = fn.random.uniform(range=[0.0, 1.0], dtype=types.FLOAT)
        crop_pos_y = fn.random.uniform(range=[0.0, 1.0], dtype=types.FLOAT)
        image = fn.crop(
            image,
            crop_h=output_size[0],
            crop_w=output_size[1],
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y,
            out_of_bounds_policy="pad",
            fill_values=0,
        )
        label = fn.crop(
            label,
            crop_h=output_size[0],
            crop_w=output_size[1],
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y,
            out_of_bounds_policy="pad",
            fill_values=255,
        )

    # color jitter
    if color_jitter_prob > 0:
        brightness = fn.random.uniform(range=color_brightness, dtype=types.FLOAT)
        contrast = fn.random.uniform(range=color_contrast, dtype=types.FLOAT)
        saturation = fn.random.uniform(range=color_saturation, dtype=types.FLOAT)
        hue = fn.random.uniform(range=color_hue, dtype=types.FLOAT)

        apply_jitter = fn.random.coin_flip(probability=color_jitter_prob)
        apply_jitter = fn.cast(apply_jitter, dtype=types.FLOAT)
        brightness = apply_jitter * brightness + (1 - apply_jitter)
        contrast = apply_jitter * contrast + (1 - apply_jitter)
        saturation = apply_jitter * saturation + (1 - apply_jitter)
        hue = apply_jitter * hue

        image = fn.brightness_contrast(image, brightness=brightness, contrast=contrast)
        image = fn.saturation(image, saturation=saturation)
        image = fn.hue(image, hue=hue)

    # horizontal flip and normalize
    mirror_coin = fn.random.coin_flip(probability=horizontal_flip_prob)
    image = fn.crop_mirror_normalize(
        image,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=(np.array(external_source.dataset.image_mean) * 255).tolist(),
        std=(np.array(external_source.dataset.image_std) * 255).tolist(),
        mirror=mirror_coin,
    )
    label = fn.flip(label, horizontal=mirror_coin)
    label = fn.squeeze(label, axis_names="C")

    return image, label, jpegs_shape, index


@pipeline_def
def semseg_validation_pipeline(
    dataset: SegmentationDataset,
    batch_size_extsrc: int,
    sampling_indices: Optional[List] = None,
    output_size: Optional[Tuple[int, int]] = None,
    num_shards: int = 1,
    shard_id: int = 0,
    random_shuffle: bool = False,
    shuffle_seed: int = 0,
):
    """Defines a DALI pipeline for deterministic semantic segmentation validation preprocessing.

    Args:
        dataset: SegmentationDataset object containing images and RGB-encoded labels
        batch_size_extsrc: Batch size for external source (must match pipeline `batch_size`)
        sampling_indices: Optional subset indices for partial evaluation
        output_size: Target output dimensions (H, W), `None` preserves original size
        num_shards: Number of shards for distributed evaluation
        shard_id: Current shards index for distributed evaluation
        random_shuffle: Enable sample shuffling (not recommended for validation)
        shuffle_seed: Seed for reproducible shuffling

    Returns:
        DALI pipeline object containing:
        - Augmented images (float32 tensor)
        - Processed segmentation masks (int32 tensor)
        - Original image shapes (for visualization)
        - Sample indices (for debugging)
    """
    external_source = SemSegExternalInputCallable(
        dataset=dataset,
        num_shards=num_shards,
        batch_size=batch_size_extsrc,
        shard_id=shard_id,
        random_shuffle=random_shuffle,
        shuffle_seed=shuffle_seed,
        sampling_indices=sampling_indices,
    )
    # Note: ``num_outputs`` is not compatible with named ``ExternalSource``
    jpegs, label, index = fn.external_source(
        source=external_source,
        num_outputs=3,
        batch=False,  # must be false
        parallel=True,
        dtype=[types.UINT8, types.UINT8, types.INT64],
    )
    jpegs_shape = fn.peek_image_shape(jpegs)
    image = fn.decoders.image(jpegs, device="mixed", output_type=types.DALIImageType.RGB)
    label = fn.decoders.image(label, device="mixed", output_type=types.DALIImageType.RGB)

    # encode RGB label
    r = fn.cast(label[:, :, 0:1], dtype=types.UINT16)
    g = fn.cast(label[:, :, 1:2], dtype=types.UINT16)
    b = fn.cast(label[:, :, 2:3], dtype=types.UINT16)
    int_label = encode_rgb888(r, g, b)
    lookup_keys = [encode_rgb888(r, g, b) for (r, g, b) in external_source.dataset.PALETTE]
    lookup_values = list(range(len(external_source.dataset.PALETTE)))
    assert len(set(lookup_keys)) == len(lookup_keys), "Duplicate key found."

    label = fn.lookup_table(int_label, default_value=255, keys=lookup_keys, values=lookup_values)

    # resize
    if output_size is not None:
        image = fn.resize(
            image,
            resize_x=output_size[1],
            resize_y=output_size[0],
            interp_type=types.INTERP_CUBIC,
        )
        label = fn.resize(
            label,
            resize_x=output_size[1],
            resize_y=output_size[0],
            interp_type=types.INTERP_NN,
        )

    # normalize
    image = fn.crop_mirror_normalize(
        image,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=(np.array(external_source.dataset.image_mean) * 255).tolist(),
        std=(np.array(external_source.dataset.image_std) * 255).tolist(),
    )
    label = fn.squeeze(label, axis_names="C")

    return image, label, jpegs_shape, index
