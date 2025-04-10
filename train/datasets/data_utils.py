"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Sequence, Tuple, Union, List, Optional, Type, Any
import torch
from torch.nn.utils.rnn import pad_sequence
import dlimp as dl
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import hashlib
from model.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


import os
import random
from typing import Callable, Optional

import numpy as np
import torch

# === Randomness ===


def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return worker_init_function if get_worker_init_fn else None


def worker_init_function(worker_id: int) -> None:
    """
    Borrowed directly from PyTorch-Lightning; inspired by this issue comment in the PyTorch repo:
        > Ref: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    Intuition: You can think of the seed sequence spawn function as a "janky" torch.Generator() or jax.PRNGKey that
    you can run iterative splitting on to get new (predictable) randomness.

    :param worker_id: Identifier for the given worker [0, num_workers) for the Dataloader in question.
    """
    # Get current `rank` (if running distributed) and `process_seed`
    global_rank, process_seed = int(os.environ["LOCAL_RANK"]), torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k) produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)


# === BFloat16 Support ===


def check_bloat16_supported() -> bool:
    try:
        import packaging.version
        import torch.cuda.nccl as nccl
        import torch.distributed as dist

        return (
            (torch.version.cuda is not None)
            and torch.cuda.is_bf16_supported()
            and (packaging.version.parse(torch.version.cuda).release >= (11, 0))
            and dist.is_nccl_available()
            and (nccl.version() >= (2, 10))
        )

    except Exception:
        return False



import math
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


# High-Fidelity Bitwise Reproduction of the LLaVa Codebase Sampler Strategy + Per-Rank Allocation Scheme (following
#   the default batching behavior of HF's Trainer Class --> derived from `accelerate`).
#
#   =>> Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L60
#   =>> Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L603
class SplitModalitySampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        modality_lengths: List[Tuple[bool, int]],
        global_batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        self.seed, self.epoch = seed, 0

        # Custom Parameters
        self.dataset, self.modality_lengths, self.drop_last = dataset, modality_lengths, drop_last
        self.global_batch_size = global_batch_size

        # For our purposes, `drop_last` is always False!
        assert not self.drop_last, "SplitModalitySampler must set `drop_last = False`!"
        self.total_size = math.ceil(len(self.dataset) / self.global_batch_size) * self.global_batch_size
        self.num_samples = self.total_size // self.num_replicas

    @staticmethod
    def reindex_batch(batch_idxs: List[int], idx2lengths: List[int], n_buckets: int) -> List[List[int]]:
        """Re-indexes a batch in a way that is conducive to DistributedSampler + grouping by seqlen per rank."""
        assert len(batch_idxs) % n_buckets == 0, "Batch length is not divisible by `num_replicas`!"

        # Establish initial buckets, capacities, and max number of elements per bucket
        n_examples_per_bucket = len(batch_idxs) // n_buckets
        bucket_indices = [[] for _ in range(n_buckets)]
        bucket_lengths = [0 for _ in range(n_buckets)]

        # Note that `batch_idxs` is already sorted by corresponding length (in descending order)
        for idx in batch_idxs:
            shortest_bucket_idx = bucket_lengths.index(min(bucket_lengths))
            bucket_indices[shortest_bucket_idx].append(idx)

            # Update `bucket_lengths` --> set length to infinity if at capacity!
            bucket_lengths[shortest_bucket_idx] += idx2lengths[idx]
            if len(bucket_indices[shortest_bucket_idx]) == n_examples_per_bucket:
                bucket_lengths[shortest_bucket_idx] = float("inf")

        return bucket_indices

    def get_modality_and_length_grouped_indices(self, generator: torch.Generator) -> List[int]:
        """
        Returns a list of indices so that each slice of `global_batch_size` consecutive indices corresponds to elements
        of the same modality with each sub-sequence of `per_replica_batch_size` (the batch size each unique device sees
        during distributed training) is roughly grouped by sequence length (for training efficiency).
        """
        multimodal_indices, multimodal_lengths = zip(
            *[(idx, length) for idx, (is_multimodal, length) in enumerate(self.modality_lengths) if is_multimodal]
        )

        # Handle Special Case --> no "unimodal" inputs
        unimodal_split = [
            (idx, length) for idx, (is_multimodal, length) in enumerate(self.modality_lengths) if not is_multimodal
        ]
        if len(unimodal_split) == 0:
            unimodal_indices, unimodal_lengths = [], []
        else:
            unimodal_indices, unimodal_lengths = zip(*unimodal_split)

        # Create a permutation of indices for each of the multimodal and unimodal data
        mm_shuffled_idxs = torch.randperm(len(multimodal_indices), generator=generator)
        uni_shuffled_idxs = torch.randperm(len(unimodal_indices), generator=generator)

        # We're going to be running sorting/grouping relative to `self.global_batch_size` and `self.num_replicas`
        g_bsz = self.global_batch_size

        # Break each of the permutations into batches of length `global_batch_size`
        mm_batch_idxs = [mm_shuffled_idxs[i : i + g_bsz].tolist() for i in range(0, len(mm_shuffled_idxs), g_bsz)]
        uni_batch_idxs = [uni_shuffled_idxs[i : i + g_bsz].tolist() for i in range(0, len(uni_shuffled_idxs), g_bsz)]

        # If "last" batch is not of length `g_bsz` --> PAD by stealing indices from the first batch!
        if len(mm_batch_idxs[-1]) < g_bsz:
            n_missing = g_bsz - len(mm_batch_idxs[-1])
            mm_batch_idxs[-1].extend(mm_batch_idxs[0][:n_missing])

        if len(uni_batch_idxs) > 0 and len(uni_batch_idxs[-1]) < g_bsz:
            n_missing = g_bsz - len(uni_batch_idxs[-1])
            uni_batch_idxs[-1].extend(uni_batch_idxs[0][:n_missing])

        # Now we're going to sort each batch by length --> this will aid in grouping by length by rank (efficiency!)
        mm_sorted_batch_idxs = [sorted(b, key=lambda i: multimodal_lengths[i], reverse=True) for b in mm_batch_idxs]
        uni_sorted_batch_idxs = [sorted(b, key=lambda i: unimodal_lengths[i], reverse=True) for b in uni_batch_idxs]

        # IMPORTANT :: At this point, for each modality, we have a list of "batches" (made up of indices) where indices
        # are sorted by example sequence length *within* each batch. To make this more concrete, consider the following:
        #   => World Size (`num_replicas`) = 2
        #   => Global Batch Size (`g_bsz`) = 4
        #   => `multimodal_indices` = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11]
        #      `multimodal_lengths` = [20, 90, 21, 22, 91, 18, 89, 19, 93, 88, 92, 17]
        #
        # At this point in the code, `mm_sorted_batch_idxs` might then look like the following (length in parenthesis):
        #   => `mm_sorted_batch_idxs`: [
        #       [4  (91), 3  (21), 0  (20), 5  (18)]    => Batch 1
        #       [6  (89), 9  (88), 7  (19), 11 (17)]    => Batch 2
        #       [8  (93), 10 (92), 1  (90), 2  (21)]    => Batch 3
        #   ]
        #
        # In practice: `g_bsz` is large (= 128), and for contiguous mini-batch "slices", length variance is low.

        # PROBLEM :: We want to split these "global batches" into equal-sized pieces, so that each "replica" (GPU)
        # sees a "mini-batch" of roughly the same sequence lengths; this is super useful for efficient training.

        # HOWEVER :: The default "access pattern" for splitting a large batch into mini-batches by a DistributedSampler
        # is akin to a "take every k" where `k` is equal to the number of replicas (GPUs) you're training on. Or, in
        # Python notation --> `rank_k_indices = flatten(mm_sorted_batch_idxs)[k::num_replicas].
        #
        # Naively translating this our example means each GPU (in our world of 2 total) sees the following indices
        # (grouped by "mini-batch" = `g_bsz / num_replicas` = 2 for convenience):
        #   => `rank_0_indices`: [ [4 (91), 0 (20)] =>> [6 (89), 7  (19)] =>> [8  (93), 1 (90)] ]
        #   => `rank_1_indices`: [ [3 (21), 5 (18)] =>> [9 (88), 11 (17)] =>> [10 (92), 2 (21)] ]
        #
        # We get lucky sometimes, but for the most part, each "mini-batch" has VASTLY DIFFERENT lengths! Bad!

        # FIX :: If we "undo" the access pattern with the following code and re-arrange the way we allocate batches
        # inside the __iter__ method below, we can allocate indices appropriately. Running the following code gives us
        # the following indices (grouped by "mini-batch" again for convenience):
        #   => `rank_0_indices`: [ [4 (91), 3 (21)] =>> [6  (89), 9 (88)] =>> [8 (93), 10 (92)] ]
        #   => `rank_1_indices`: [ [5 (18), 0 (20)] =>> [11 (17), 7 (19)] =>> [2 (21),  1 (90)] ]
        #
        # Much better! As `g_bsz` and `dataset` grow, we're more often than not getting *decent* groupings!
        mm_length_bucketed_idxs = [
            self.reindex_batch(batch, multimodal_lengths, self.num_replicas) for batch in mm_sorted_batch_idxs
        ]
        uni_length_bucketed_idxs = [
            self.reindex_batch(batch, unimodal_lengths, self.num_replicas) for batch in uni_sorted_batch_idxs
        ]

        # Note :: Because of the initial `randperm` --> we're indexing both sets from 0 (we're clobbering the range)
        #   => Flatten indices --> index into original `{modality}_indices` then re-batch!
        mm_output_idxs = [idx for batch in mm_length_bucketed_idxs for bucket in batch for idx in bucket]
        mm_reindexed = [multimodal_indices[idx] for idx in mm_output_idxs]
        mm_batches = [mm_reindexed[i : i + g_bsz] for i in range(0, len(mm_reindexed), g_bsz)]

        uni_output_idxs = [idx for batch in uni_length_bucketed_idxs for bucket in batch for idx in bucket]
        uni_reindexed = [unimodal_indices[idx] for idx in uni_output_idxs]
        uni_batches = [uni_reindexed[i : i + g_bsz] for i in range(0, len(uni_reindexed), g_bsz)]

        # Finally, randomly permute the multimodal & unimodal batches, merging into a single stream of indices
        merged_batches = mm_batches + uni_batches
        merge_idxs = torch.randperm(len(merged_batches), generator=generator)
        all_batches = [merged_batches[idx] for idx in merge_idxs]

        # [Quality of Life] Shift "max length" batch to index 0 --> if we OOM, it happens immediately!
        all_lengths = [length + ((_n_patches := 24 * 24) if is_mm else 0) for is_mm, length in self.modality_lengths]
        all_batches_max_lengths = []
        for batch in all_batches:
            all_batches_max_lengths.append(max([all_lengths[idx] for idx in batch]))

        # Identify Batch with "max length" --> Swap into Index 0
        longest_batch_idx = np.argmax(all_batches_max_lengths)
        all_batches[0], all_batches[longest_batch_idx] = all_batches[longest_batch_idx], all_batches[0]

        # Flatten & Return all Indices
        indices = [idx for batch in all_batches for idx in batch]
        return indices

    def __iter__(self) -> Iterator:
        """Deterministically shuffle, then split indices by modality and length."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = self.get_modality_and_length_grouped_indices(g)
        assert len(set(indices)) == len(self.modality_lengths) == len(self.dataset), "Oops!"
        assert (len(indices) % self.global_batch_size == 0) and (len(indices) % self.num_replicas) == 0, "Oops"

        # Note :: We compute per-replica batch size as a function of `global_batch` and `num_replicas` to ensure that
        # gradient accumulation doesn't affect what indices are assigned a given rank.
        per_replica_batch_size = self.global_batch_size // self.num_replicas

        # Tensorize & Unravel --> rather than yielding via a `take_every` --> we want to partition a global batch
        # across replicas by assigning each a contiguous sub-sequence.
        indices_t = torch.as_tensor(indices)
        per_replica_batch_indices_t = indices_t.reshape(-1, per_replica_batch_size)
        replica_indices_t = per_replica_batch_indices_t[self.rank :: self.num_replicas]

        replica_indices = replica_indices_t.flatten().tolist()
        return iter(replica_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """To be called *between* epochs, prior to DataLoader instantiation; ensures random order across epochs."""
        self.epoch = epoch


# ruff: noqa: B023
def augment(obs: Dict, seed: tf.Tensor, augment_kwargs: Union[Dict, Dict[str, Dict]]) -> Dict:
    """Augments images, skipping padding images."""
    image_names = {key[6:] for key in obs if key.startswith("image_")}

    # "augment_order" is required in augment_kwargs, so if it's there, we can assume that the user has passed
    # in a single augmentation dict (otherwise, we assume that the user has passed in a mapping from image
    # name to augmentation dict)
    if "augment_order" in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    for i, name in enumerate(image_names):
        if name not in augment_kwargs:
            continue
        kwargs = augment_kwargs[name]
        logging.debug(f"Augmenting image_{name} with kwargs {kwargs}")
        obs[f"image_{name}"] = tf.cond(
            obs["pad_mask_dict"][f"image_{name}"],
            lambda: dl.transforms.augment_image(
                obs[f"image_{name}"],
                **kwargs,
                seed=seed + i,  # augment each image differently
            ),
            lambda: obs[f"image_{name}"],  # skip padding images
        )

    return obs


def decode_and_resize(
    obs: Dict,
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
) -> Dict:
    """Decodes images and depth images, and then optionally resizes them."""
    image_names = {key[6:] for key in obs if key.startswith("image_")}
    depth_names = {key[6:] for key in obs if key.startswith("depth_")}

    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        if name not in resize_size:
            logging.warning(
                f"No resize_size was provided for image_{name}. This will result in 1x1 "
                "padding images, which may cause errors if you mix padding and non-padding images."
            )
        image = obs[f"image_{name}"]
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                # this is a padding image
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
            else:
                image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
        elif image.dtype != tf.uint8:
            raise ValueError(f"Unsupported image dtype: found image_{name} with dtype {image.dtype}")
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f"image_{name}"] = image

    for name in depth_names:
        if name not in depth_resize_size:
            logging.warning(
                f"No depth_resize_size was provided for depth_{name}. This will result in 1x1 "
                "padding depth images, which may cause errors if you mix padding and non-padding images."
            )
        depth = obs[f"depth_{name}"]

        if depth.dtype == tf.string:
            if tf.strings.length(depth) == 0:
                depth = tf.zeros((*depth_resize_size.get(name, (1, 1)), 1), dtype=tf.float32)
            else:
                depth = tf.io.decode_image(depth, expand_animations=False, dtype=tf.float32)[..., 0]
        elif depth.dtype != tf.float32:
            raise ValueError(f"Unsupported depth dtype: found depth_{name} with dtype {depth.dtype}")

        if name in depth_resize_size:
            depth = dl.transforms.resize_depth_image(depth, size=depth_resize_size[name])

        obs[f"depth_{name}"] = depth

    return obs



def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # if no absolute_action_mask was provided, assume all actions are relative
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )

    # actions past the goal timestep become neutral
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj


# ruff: noqa: B023
def normalize_action_and_proprio(traj: Dict, metadata: Dict, normalization_type: NormalizationType):
    """Normalizes the action and proprio fields of a trajectory using the given metadata."""
    keys_to_normalize = {"action": "action", "proprio": "observation/proprio"}

    if normalization_type == NormalizationType.NORMAL:
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x),
            )

        return traj

    elif normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:
        for key, traj_key in keys_to_normalize.items():
            if normalization_type == NormalizationType.BOUNDS:
                low = metadata[key]["min"]
                high = metadata[key]["max"]
            elif normalization_type == NormalizationType.BOUNDS_Q99:
                low = metadata[key]["q01"]
                high = metadata[key]["q99"]
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1),
                    x,
                ),
            )

            # Note (Moo Jin): Map unused action dimensions (i.e., dimensions where min == max) to all 0s.
            zeros_mask = metadata[key]["min"] == metadata[key]["max"]
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == traj_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            )

        return traj

    raise ValueError(f"Unknown Normalization Type {normalization_type}")


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32), lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return 1 - actions

# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]) -> None:
    print("\n######################################################################################")
    print(f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print("######################################################################################\n")


def get_dataset_statistics(
    dataset: dl.DLataset,
    hash_dependencies: Tuple[str, ...],
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Either computes the statistics of a dataset or loads them from a cache file if this function has been called before
    with the same `hash_dependencies`.

    Currently, the statistics include the min/max/mean/std of the actions and proprio as well as the number of
    transitions and trajectories in the dataset.
    """
    unique_hash = hashlib.sha256("".join(hash_dependencies).encode("utf-8"), usedforsecurity=False).hexdigest()

    # Fallback local path for when data_dir is not writable or not provided
    local_path = os.path.expanduser(os.path.join("~", ".cache", "orca", f"dataset_statistics_{unique_hash}.json"))
    if save_dir is not None:
        path = tf.io.gfile.join(save_dir, f"dataset_statistics_{unique_hash}.json")
    else:
        path = local_path

    # check if cache file exists and load
    if tf.io.gfile.exists(path):
        overwatch.info(f"Loading existing dataset statistics from {path}.")
        with tf.io.gfile.GFile(path, "r") as f:
            metadata = json.load(f)
        return metadata

    if os.path.exists(local_path):
        overwatch.info(f"Loading existing dataset statistics from {local_path}.")
        with open(local_path, "r") as f:
            metadata = json.load(f)
        return metadata

    dataset = dataset.traj_map(
        lambda traj: {
            "action": traj["action"],
            "proprio": (
                traj["observation"]["proprio"] if "proprio" in traj["observation"] else tf.zeros_like(traj["action"])
            ),
        }
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    overwatch.info("Computing dataset statistics. This may take a bit, but should only need to happen once.")
    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    for traj in tqdm(dataset.iterator(), total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None):
        actions.append(traj["action"])
        proprios.append(traj["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1

    actions, proprios = np.concatenate(actions), np.concatenate(proprios)
    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
            "q01": np.quantile(actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        },
        "proprio": {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
            "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
            "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    try:
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
    except tf.errors.PermissionDeniedError:
        overwatch.warning(f"Could not write dataset statistics to {path}. Writing to {local_path} instead.")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(metadata, f)

    return metadata


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    overwatch.info(f"Saved dataset statistics file at path {out_path}")


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """
    Allocates an integer number of threads across datasets based on weights.

    The final array sums to `n`, but each element is no less than 1. If `n` is None, then every dataset is assigned a
    value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)

        # Recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1

    return allocation
