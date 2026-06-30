"""Shared DataLoader factories for Q-Former alignment pipelines.

Used by both stage 1 representation training and stage 2 generative
pretraining so the dataset/collate/loader plumbing stays in one place.
"""

import os
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset


FilterFn = Callable[[QFormerAlignmentDataset], Dataset]


class BalancedTypeBatchSampler(Sampler):
    """CHANGE 2f: guarantee ``min_item_text`` ``item_text`` samples per batch.

    Stage-1 mixes ``item_text`` (~one per item), ``item_item`` and (capped)
    ``user_item`` samples in one flat list. The ITC/ITM/ITG objectives only fire
    on the ``item_text`` rows in a batch, and their in-batch negatives ARE the
    other ``item_text`` rows — so with a plain shuffled loader most batches carry
    too few of them and the alignment objective is starved (see config note).

    This sampler builds each batch from ``min_item_text`` item-text rows (drawn
    from a reshuffled, cycled pool, so the small item-text set is oversampled)
    plus the remaining slots from the other types (each seen once per epoch).
    The number of batches is set so every non-item-text sample appears once.
    """

    def __init__(
        self,
        sample_types: List[str],
        batch_size: int,
        min_item_text: int,
        shuffle: bool = True,
        seed: int = 0,
        item_text_type: str = "item_text",
    ) -> None:
        self.batch_size = int(batch_size)
        self.min_item_text = max(0, min(int(min_item_text), self.batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        self.item_text_idx = [i for i, t in enumerate(sample_types) if t == item_text_type]
        self.other_idx = [i for i, t in enumerate(sample_types) if t != item_text_type]
        if not self.item_text_idx:
            # Nothing to balance; fall back to a single flat pool.
            self.item_text_idx, self.other_idx = [], list(range(len(sample_types)))
            self.min_item_text = 0

        self.other_per_batch = max(1, self.batch_size - self.min_item_text)
        if self.other_idx:
            self._num_batches = max(1, (len(self.other_idx) + self.other_per_batch - 1) // self.other_per_batch)
        else:
            total = max(1, len(self.item_text_idx))
            self._num_batches = max(1, (total + self.batch_size - 1) // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches

    def _generator(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return g

    @staticmethod
    def _perm(items, g):
        if not items:
            return []
        order = torch.randperm(len(items), generator=g).tolist()
        return [items[i] for i in order]

    def __iter__(self):
        g = self._generator() if self.shuffle else None

        def _maybe_shuffle(items):
            return self._perm(items, g) if self.shuffle else list(items)

        other_pool = _maybe_shuffle(self.other_idx)
        item_text_pool = _maybe_shuffle(self.item_text_idx)
        it_ptr = 0

        def _next_item_text(n):
            nonlocal it_ptr, item_text_pool
            out = []
            if not item_text_pool:
                return out
            while len(out) < n:
                if it_ptr >= len(item_text_pool):
                    item_text_pool = _maybe_shuffle(self.item_text_idx)  # reshuffle + cycle
                    it_ptr = 0
                out.append(item_text_pool[it_ptr])
                it_ptr += 1
            return out

        for b in range(self._num_batches):
            batch = list(other_pool[b * self.other_per_batch:(b + 1) * self.other_per_batch]) if self.other_idx else []
            need = self.batch_size - len(batch)
            batch.extend(_next_item_text(need))
            if self.shuffle and g is not None and len(batch) > 1:
                order = torch.randperm(len(batch), generator=g).tolist()
                batch = [batch[i] for i in order]
            yield batch


def qformer_collate(batch):
    """Stack tensors, keep other fields as lists."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


def build_qformer_loader(
    cfg,
    filename: str,
    shuffle: bool,
    filter_fn: Optional[FilterFn] = None,
    balanced: bool = False,
    min_item_text: int = 0,
    seed: int = 0,
) -> DataLoader:
    """Build a single Q-Former DataLoader from a samples pickle.

    Parameters
    ----------
    cfg
        Config namespace with ``batch_size`` and ``num_workers`` attributes.
    filename
        Path to a ``.pkl`` file produced by ``QFormerAlignmentBuilder``.
    shuffle
        Whether to shuffle the loader.
    filter_fn
        Optional callable that receives the loaded ``QFormerAlignmentDataset``
        and returns a (possibly subsetted) ``Dataset``. Use this in stage 2
        to keep only ``item_text`` samples.
    balanced
        CHANGE 2f: when True (and the dataset is not pre-filtered to a single
        type), use ``BalancedTypeBatchSampler`` to guarantee ``min_item_text``
        item-text samples per batch. Intended for the Stage-1 *train* loader.
    """
    dataset: Dataset = QFormerAlignmentDataset(filename=filename)
    if filter_fn is not None:
        dataset = filter_fn(dataset)

    if balanced and min_item_text > 0 and hasattr(dataset, "samples"):
        sample_types = [s["sample_type"] for s in dataset.samples]
        batch_sampler = BalancedTypeBatchSampler(
            sample_types,
            batch_size=int(cfg.batch_size),
            min_item_text=int(min_item_text),
            shuffle=shuffle,
            seed=int(seed),
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=qformer_collate,
            num_workers=int(cfg.num_workers),
        )

    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        collate_fn=qformer_collate,
        num_workers=int(cfg.num_workers),
    )


def build_qformer_loaders(
    cfg,
    data_dir: str,
    train_filename: str = "train_qformer_ood2.pkl",
    val_filename: str = "valid_qformer_ood2.pkl",
    test_filename: Optional[str] = "test_qformer_ood2.pkl",
    filter_fn: Optional[FilterFn] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train/val/test loaders for the Q-Former alignment task.

    Pass ``test_filename=None`` if the calling stage has no test split
    (stage 2 generative pretraining); the returned ``test_loader`` is
    ``None`` in that case.

    CHANGE 2f: balanced batching (``cfg.balanced_batches`` +
    ``cfg.min_item_text_per_batch``) is applied to the TRAIN loader only, so
    validation/test keep the natural distribution.
    """
    balanced = bool(cfg.get("balanced_batches", False)) if hasattr(cfg, "get") else False
    min_item_text = int(cfg.get("min_item_text_per_batch", 0)) if hasattr(cfg, "get") else 0
    seed = int(cfg.get("seed", 0)) if hasattr(cfg, "get") else 0

    train_loader = build_qformer_loader(
        cfg,
        filename=os.path.join(data_dir, train_filename),
        shuffle=True,
        filter_fn=filter_fn,
        balanced=balanced,
        min_item_text=min_item_text,
        seed=seed,
    )
    val_loader = build_qformer_loader(
        cfg, filename=os.path.join(data_dir, val_filename), shuffle=False, filter_fn=filter_fn
    )
    test_loader = None
    if test_filename is not None:
        test_loader = build_qformer_loader(
            cfg, filename=os.path.join(data_dir, test_filename), shuffle=False, filter_fn=filter_fn
        )
    return train_loader, val_loader, test_loader
