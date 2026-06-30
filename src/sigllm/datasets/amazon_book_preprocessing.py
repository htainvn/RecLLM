"""Amazon-Book preprocessing: convert a CoLLM / SeLLa-Rec Amazon-Book split
into this repo's ``*_ood2.pkl`` schema.

Why a converter (not a from-raw builder)
----------------------------------------
SeLLa-Rec (arXiv:2504.10107) builds on CoLLM (arXiv:2310.19488), which released
a *preprocessed* Amazon-Book split (rating>=4 -> positive, year-2017 slice,
20-core user/item filtering). Reusing that split keeps the comparison against
the SeLLa-Rec baseline apples-to-apples instead of re-deriving filtering/splits
ourselves. This module takes the CoLLM files (whatever exact column names they
ship with) and emits the files ``build_ml1m`` would have produced, so every
downstream component (``MovieOODDataset``, the Q-Former alignment builder, MF,
Stage 1-3) runs unchanged on Amazon.

Target schema (identical to ``data_preprocessing.build_ml1m``)
--------------------------------------------------------------
Columns: ``uid, iid, label, timestamp, his, his_title, title, genres, flag,
not_cold``. Files written to ``out_dir``:

- ``train_ood2.pkl`` (flag = -1)
- ``valid_ood2.pkl`` (flag = 0)
- ``test_ood2.pkl``  (flag = 1)
- ``valid_small_ood2.pkl`` (50% sample of valid)
- ``users_map.pkl`` / ``items_map.pkl`` (raw-id -> contiguous-id, 0 = padding)

Run ``process_warm_cold`` afterwards (unchanged) to add warm/cold tags, then
``build_qformer_dataset`` to produce the Q-Former pickles.

Because the exact CoLLM column names vary by release, every field is mapped
through ``ColumnMap`` and you can inspect any file first with
``inspect_collm_files``. The defaults match the most common CoLLM layout; the
notebook prints the real schema so the mapping can be adjusted in one place.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sigllm.common import NotebookLogger
from sigllm.datasets.data_preprocessing import deal_with_each_u

LOGGER = NotebookLogger.rich_logger("sigllm.amazon_book_prep")


def log_step(title: str, detail: Optional[str] = None) -> None:
    LOGGER.info(title if detail is None else f"{title} | {detail}")


@dataclass
class ColumnMap:
    """Maps source (CoLLM/SeLLa-Rec) column names to the canonical names.

    Set a field to the source column name, or leave it ``None`` if the source
    does not provide it (it will be derived where possible). ``rating`` is only
    used when ``label`` is absent (binarized with ``rating_threshold``).
    """

    uid: str = "uid"
    iid: str = "iid"
    label: Optional[str] = "label"
    rating: Optional[str] = None
    timestamp: Optional[str] = "timestamp"
    his: Optional[str] = None          # history item-id list, if already present
    his_title: Optional[str] = None    # history title list, if already present
    title: Optional[str] = None        # item title, if carried on interactions
    genres: Optional[str] = None       # item categories, if carried on interactions


def _read_any(path: str) -> pd.DataFrame:
    """Load a tabular file by extension (.pkl/.csv/.parquet/.json[l])."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".pickle"):
        obj = pd.read_pickle(path)
        return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".json", ".jsonl"):
        return pd.read_json(path, lines=(ext == ".jsonl"))
    raise ValueError(f"Unsupported file extension for {path!r}")


def inspect_collm_files(paths: dict) -> dict:
    """Print columns/dtypes/head for each provided file; return {name: df}.

    Use this BEFORE ``build_amazon_book`` to discover the real column names and
    fill in ``ColumnMap`` accordingly. ``paths`` maps a label -> file path,
    e.g. ``{"train": ".../train.pkl", "item_text": ".../item2text.csv"}``.
    """
    frames = {}
    for name, path in paths.items():
        if path is None or not os.path.exists(path):
            log_step("inspect: missing", f"{name} -> {path}")
            continue
        df = _read_any(path)
        frames[name] = df
        log_step(f"inspect: {name}", f"path={path} shape={df.shape}")
        log_step(f"  columns", ", ".join(map(str, df.columns)))
        log_step(f"  dtypes", ", ".join(f"{c}:{t}" for c, t in df.dtypes.items()))
        try:
            log_step(f"  head[0]", str(df.iloc[0].to_dict()))
        except Exception:  # pragma: no cover - defensive on odd frames
            pass
    return frames


def _standardize(df: pd.DataFrame, cmap: ColumnMap, rating_threshold: float) -> pd.DataFrame:
    """Rename source columns to canonical names and derive label/timestamp."""
    out = df.copy()
    rename = {}
    for canonical in ("uid", "iid", "label", "timestamp", "his", "his_title", "title", "genres"):
        src = getattr(cmap, canonical)
        if src is not None and src in out.columns and src != canonical:
            rename[src] = canonical
    out = out.rename(columns=rename)

    if "label" not in out.columns:
        if cmap.rating is None or cmap.rating not in df.columns:
            raise KeyError(
                "No 'label' column and no usable 'rating' column to binarize. "
                "Set ColumnMap.label or ColumnMap.rating."
            )
        out["label"] = (df[cmap.rating].astype(float) >= rating_threshold).astype(int)

    if "timestamp" not in out.columns:
        # No time -> synthesize a stable per-row order so history is deterministic.
        out["timestamp"] = np.arange(len(out), dtype=np.int64)

    for required in ("uid", "iid"):
        if required not in out.columns:
            raise KeyError(f"Required column '{required}' missing after mapping (cmap.{required}={getattr(cmap, required)!r}).")
    return out


def _load_item_text(
    item_text_path: Optional[str],
    item_text_map: ColumnMap,
) -> tuple[dict, dict]:
    """Return (iid -> title, iid -> 'cat|cat' genres) from an item-text file.

    ``item_text_map`` reuses ColumnMap purely for its ``iid``/``title``/``genres``
    fields. Returns empty dicts when ``item_text_path`` is None (titles must
    then be carried on the interaction frames themselves).
    """
    if item_text_path is None:
        return {}, {}
    meta = _read_any(item_text_path)
    iid_col = item_text_map.iid
    title_col = item_text_map.title or "title"
    genres_col = item_text_map.genres
    if iid_col not in meta.columns or title_col not in meta.columns:
        raise KeyError(
            f"Item-text file must contain id column '{iid_col}' and title column "
            f"'{title_col}'. Found: {list(meta.columns)}"
        )

    def _norm_genres(value) -> str:
        if genres_col is None or value is None:
            return ""
        if isinstance(value, (list, tuple, np.ndarray)):
            return "|".join(str(v).strip() for v in value if str(v).strip())
        return str(value).strip()

    title_lookup, genre_lookup = {}, {}
    for row in meta.itertuples(index=False):
        rid = int(getattr(row, iid_col))
        title_lookup[rid] = str(getattr(row, title_col))
        genre_lookup[rid] = _norm_genres(getattr(row, genres_col)) if genres_col else ""
    return title_lookup, genre_lookup


def build_amazon_book(
    train_path: str,
    valid_path: str,
    test_path: str,
    out_dir: str,
    item_text_path: Optional[str] = None,
    cmap: Optional[ColumnMap] = None,
    item_text_map: Optional[ColumnMap] = None,
    rating_threshold: float = 4.0,
    reuse_ids: bool = False,
    seed: int = 2023,
) -> tuple:
    """Convert a CoLLM/SeLLa-Rec Amazon-Book split into ``*_ood2.pkl`` files.

    Parameters
    ----------
    train_path, valid_path, test_path
        CoLLM interaction split files (pkl/csv/parquet/json).
    out_dir
        Destination directory for the ``*_ood2.pkl`` artifacts.
    item_text_path
        Optional item-metadata file mapping item id -> title (+ categories).
        Required only if ``title`` is not already carried on the interaction
        frames (``cmap.title``).
    cmap, item_text_map
        Source-column mappings (see ``ColumnMap``). Defaults assume canonical
        names; inspect first with ``inspect_collm_files`` and override.
    rating_threshold
        Used only when interaction frames carry ``rating`` but not ``label``.
    reuse_ids
        If True, keep the source ids verbatim (assumes they are already
        contiguous with 0 reserved for padding). If False (default), remap to a
        fresh contiguous 1..N space like ``build_ml1m``.
    """
    cmap = cmap or ColumnMap()
    item_text_map = item_text_map or ColumnMap()

    log_step("[1/8] Load splits", f"train={train_path}, valid={valid_path}, test={test_path}")
    train = _standardize(_read_any(train_path), cmap, rating_threshold)
    valid = _standardize(_read_any(valid_path), cmap, rating_threshold)
    test = _standardize(_read_any(test_path), cmap, rating_threshold)

    train["flag"] = -1
    valid["flag"] = 0
    test["flag"] = 1
    log_step("Split sizes", f"train={len(train):,}, valid={len(valid):,}, test={len(test):,}")

    # ----- item text (title + categories/genres) -----
    log_step("[2/8] Item text", f"item_text_path={item_text_path}")
    title_lookup, genre_lookup = _load_item_text(item_text_path, item_text_map)
    has_inline_title = "title" in train.columns

    data = pd.concat([train, valid, test], axis=0, ignore_index=True)

    if not has_inline_title:
        if not title_lookup:
            raise KeyError(
                "No inline 'title' column and no item_text_path provided; cannot "
                "attach item text. Set cmap.title or pass item_text_path."
            )
        data["title"] = data["iid"].map(lambda x: title_lookup.get(int(x), str(x)))
        data["genres"] = data["iid"].map(lambda x: genre_lookup.get(int(x), ""))
    else:
        if "genres" not in data.columns:
            data["genres"] = data["iid"].map(lambda x: genre_lookup.get(int(x), "")) if genre_lookup else ""

    data["title"] = data["title"].fillna("").astype(str)
    data["genres"] = data["genres"].fillna("").astype(str)

    # ----- id remap (mirror build_ml1m) -----
    log_step("[3/8] Remap ids", f"reuse_ids={reuse_ids}")
    if reuse_ids:
        users_map = {int(u): int(u) for u in data["uid"].unique()}
        items_map = {int(i): int(i) for i in data["iid"].unique()}
        users_map[0] = 0
        items_map[0] = 0
    else:
        users = data["uid"].unique()
        items = data["iid"].unique()
        users_map = dict(zip(users, np.arange(users.shape[0]) + 1))
        items_map = dict(zip(items, np.arange(items.shape[0]) + 1))
        users_map[0] = 0
        items_map[0] = 0
        data["uid"] = data["uid"].map(users_map)
        data["iid"] = data["iid"].map(items_map)

    # ----- history (reuse source his, else derive sequentially) -----
    has_history = "his" in data.columns and data["his"].notna().any()
    if has_history:
        log_step("[4/8] History", "using source 'his' column")
        if not reuse_ids:
            data["his"] = data["his"].apply(
                lambda seq: [items_map.get(int(k), 0) for k in (seq if isinstance(seq, (list, tuple, np.ndarray)) else [])]
            )
        if "his_title" not in data.columns:
            id2title = {items_map.get(int(k), 0): v for k, v in title_lookup.items()} if (title_lookup and not reuse_ids) else title_lookup
            data["his_title"] = data["his"].apply(
                lambda seq: [id2title.get(int(k), "") for k in seq]
            )
    else:
        log_step("[4/8] History", "deriving sequentially (deal_with_each_u)")
        data = data.sort_values(by=["uid", "timestamp"])
        grouped = data.groupby("uid").agg(
            {"iid": list, "label": list, "title": list, "genres": list, "timestamp": list, "flag": list}
        )
        results = []
        for u in grouped.index:
            results.extend(deal_with_each_u(grouped.loc[u], u))
        cols = ["uid", "iid", "timestamp", "his", "his_title", "title", "label", "genres", "flag"]
        data = pd.DataFrame(results, columns=cols)
        data["his"] = data["his"].apply(lambda x: [int(k) for k in x])

    # ----- split back, cold-start flags, serialize -----
    log_step("[5/8] Split back")
    train_ = data[data["flag"] == -1].copy()
    valid_ = data[data["flag"] == 0].copy()
    test_ = data[data["flag"] == 1].copy()

    log_step("[6/8] Cold-start flags", "not_cold = uid & iid both seen in train")
    train_users = set(train_["uid"].unique())
    train_items = set(train_["iid"].unique())
    valid_["not_cold"] = (valid_["uid"].isin(train_users) & valid_["iid"].isin(train_items)).astype("int")
    test_["not_cold"] = (test_["uid"].isin(train_users) & test_["iid"].isin(train_items)).astype("int")
    train_["not_cold"] = 1

    log_step("[7/8] Serialize", f"out_dir={out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    train_.to_pickle(os.path.join(out_dir, "train_ood2.pkl"))
    valid_.to_pickle(os.path.join(out_dir, "valid_ood2.pkl"))
    test_.to_pickle(os.path.join(out_dir, "test_ood2.pkl"))
    valid_.sample(frac=0.5, random_state=seed).to_pickle(os.path.join(out_dir, "valid_small_ood2.pkl"))
    with open(os.path.join(out_dir, "users_map.pkl"), "wb") as f:
        pickle.dump(users_map, f)
    with open(os.path.join(out_dir, "items_map.pkl"), "wb") as f:
        pickle.dump(items_map, f)

    user_num = int(data["uid"].max()) + 1
    item_num = int(data["iid"].max()) + 1
    log_step(
        "[8/8] Done",
        f"users={user_num - 1:,} (user_num={user_num}), items={item_num - 1:,} "
        f"(item_num={item_num}), train={len(train_):,}, valid={len(valid_):,}, "
        f"test={len(test_):,}, pos_rate(train)={train_['label'].mean():.3f}",
    )
    log_step("Set in config", f"rec_config.user_num={user_num}, rec_config.item_num={item_num}")
    return train_, valid_, test_, users_map, items_map
