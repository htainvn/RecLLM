"""Builder for the Amazon-Book dataset.

Amazon-Book uses the exact same ``*_ood2.pkl`` schema as ML-1M (produced by
``amazon_book_preprocessing.build_amazon_book``), so it reuses the generic
``MovieOODDataset`` reader unchanged and only needs a distinct registry name so
configs can select it via ``datasets: { amazon_book: {...} }``.
"""

from __future__ import annotations

from sigllm.common.registry import registry
from sigllm.datasets.base.rec_base_dataset_builder import RecBaseDatasetBuilder
from sigllm.datasets.movie.movie_ood_dataset import MovieOODDataset


@registry.register_builder("amazon_book")
class AmazonBookBuilder(RecBaseDatasetBuilder):
    """Construct Amazon-Book dataset splits (schema-compatible with MovieOOD)."""

    train_dataset_cls = MovieOODDataset
