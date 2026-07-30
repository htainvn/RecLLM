from __future__ import annotations

import os
import random

import pandas as pd
import torch

from sigllm.datasets.base.rec_base_dataset_builder import RecBaseDatasetBuilder
from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset


"""Builder for Q-Former phase-1 alignment data.

This builder creates a flat, ILM-style dataset from the sequential ML-1M
preprocessed splits. The output is a torch-saved dictionary:

    {
        "seed": int,
        "samples": list[dict],
        "stats": dict,
    }

Each entry in "samples" has the same schema regardless of objective:

    sample_type:
        One of "item_text", "item_item", "user_item". ILM-style training uses
        "item_text" and "item_item"; "user_item" is kept as an optional
        experiment for older checkpoints.

    u:
        User id. It is meaningful only for "user_item" samples. For item-only
        objectives it is set to 0, the padding/dummy id.

    i_left:
        Main item id. For "item_text", this is the item aligned to metadata
        text. For "item_item", this is the left item in a positive co-watch
        pair. For "user_item", this is the positive item liked by user u.

    i_right:
        Secondary item id. It is meaningful only for "item_item" samples.
        For "item_text" and "user_item", it is set to 0.

    text:
        Natural-language item metadata used by the item-text objective. It is
        populated for "item_text" and kept as useful metadata for "user_item".
        "item_item" uses collaborative pairs only, so text is empty there.

    instruction:
        A short task prompt encoded by the frozen text encoder and passed into
        Q-Former as conditioning tokens. These are intentionally fixed/simple
        templates for the first clean baseline.

    weight:
        Sample weight metadata. Currently the trainer does not apply weighted
        loss, but item-item pairs store co-occurrence counts here so the data
        file preserves pair strength for later sampling/weighting experiments.
"""
# @registry.register_builder("qformer_alignment")
class QFormerAlignmentBuilder(RecBaseDatasetBuilder):
    """Construct ILM-style Q-Former alignment samples."""

    TEMPL_ITEM_TEXT = [
        "Represent this movie for recommendation using its title and genres.",
        "Align this movie metadata with its collaborative filtering representation.",
        "Given the movie metadata, extract recommendation-relevant item features.",
        "Use the title and genres to describe this movie in the item embedding space.",
        "Map this movie's textual attributes to its collaborative recommendation signal.",
        "Identify the movie preferences implied by its title and genre metadata.",
        "Create a language-aligned representation of this movie for recommendation.",
        "Summarize this movie as an item a recommender system can compare.",
        "Based on the title and genres, represent what kind of users may like this movie.",
        "Encode the semantic information of this movie for item-language alignment.",
        "Use a few metadata cues to align this movie with behavioral item signals.",
        "Produce a recommendation-aware representation from this movie description.",
    ]

    TEMPL_ITEM_ITEM = [
        "Align movies that appear close together in positive user histories.",
        "Represent these two movies as behaviorally related items.",
        "Given user interaction patterns, pull these related movies closer together.",
        "Align two movies that are likely to share audience preferences.",
        "Use collaborative behavior to represent these movies as similar items.",
        "Compare these co-watched movies in the recommendation embedding space.",
        "Learn item features that preserve this positive item-item relationship.",
        "Encode the behavioral connection between these two movies.",
        "Represent this movie pair using shared recommendation signals.",
        "Use co-occurrence evidence to align the two movie representations.",
    ]

    TEMPL_USER_ITEM = [
        "Align this user with a movie they liked.",
        "Represent a positive user-movie interaction for recommendation.",
    ]

    train_dataset_cls = QFormerAlignmentDataset

    @staticmethod
    def _item_text_bucket(iid: int, split_seed: int, valid_frac: float, test_frac: float) -> str:
        """Deterministic item-level split for the item_text objective.

        Depends only on ``(split_seed, iid)`` — NOT on which split's dataframe
        is being processed — so an item lands in exactly one bucket across the
        train/valid/test builder runs.
        """
        r = random.Random(f"{split_seed}:{int(iid)}").random()
        if r < valid_frac:
            return "valid"
        if r < valid_frac + test_frac:
            return "test"
        return "train"

    @staticmethod
    def build_qformer_alignment_samples(
        input_pkl_path: str,
        output_path: str,
        seed: int = 42,
        item_pair_window: int = 2,
        max_item_item_pairs: int | None = None,
        max_user_item_pairs: int | None = None,
        include_user_item: bool = False,
        max_history_length: int = 10,
        item_text_split: str = "train",
        item_text_valid_frac: float = 0.0,
        item_text_test_frac: float = 0.0,
        item_text_split_seed: int = 42,
        min_positive_history: int = 1,
    ):
        """See the module docstring for the sample schema.

        ``max_history_length`` caps the per-row ``his`` list of user_item
        samples. It must be the SAME value Stage 3's ``MovieOODDataset`` uses
        (``datasets.*.max_history_length``): Stage 1 pretrains the history
        pooling on sequences up to this length, and pretraining on L=50 while
        Stage 3 runs L=10 (or vice versa) trains the pooling on a length
        regime it never sees again.

        ``item_text_valid_frac`` / ``item_text_test_frac`` carve a
        deterministic item-level holdout for the item_text objective: an item's
        (CF, text) pair appears in exactly ONE split's item_text block. Without
        this, every split got item_text for every item it contains, and since
        the interaction split is temporal, nearly all val/test items also occur
        in train — ITC/ITM/ITG on val then measured memorization, not
        generalization. Both fracs 0.0 = legacy behaviour (all items in every
        split).

        ``min_positive_history`` must mirror
        ``datasets.*.min_positive_history``, the SeLLa-parity filter
        ``MovieOODDataset`` applies. Stage 3 never sees rows with a shorter
        history, so building from the unfiltered frame trained Stage 1 on
        user_item rows — and on the item_item pairs and item catalog derived
        from them — that no downstream stage ever encounters. 1 = no filter
        (legacy behaviour).
        """
        rng = random.Random(seed)

        # Input contract after data_preprocessing.build_ml1m():
        #
        #   uid  iid  label  timestamp  his           title             genres
        #   1    10   1      ...        [0]           Toy Story         Animation|Children|Comedy
        #   1    25   0      ...        [0, 10]       Jumanji           Adventure|Children|Fantasy
        #   1    33   1      ...        [0, 10]       Grumpier Old Men  Comedy|Romance
        #
        # History grows only after positive interactions, so the positive rows
        # are enough to recover user-item pairs and local item-item co-watch
        # pairs without using explicit negative samples.
        df = pd.read_pickle(input_pkl_path).reset_index(drop=True)

        # Fail fast if an old artifact is passed in. The Q-Former phase-1 data
        # depends on title + genres for item-text alignment and timestamp + uid
        # for sequence-aware item-item pair construction.
        required_columns = {"uid", "iid", "label", "timestamp", "his", "title", "genres"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise KeyError(
                "Q-Former alignment builder requires preprocessed data with columns: "
                f"{sorted(required_columns)}. Missing: {missing_columns}"
            )

        # SeLLa-parity history filter. It applies to the INTERACTION ROWS only —
        # the user_item samples and the item_item co-watch pairs derived from
        # them — because that is what Stage 3's MovieOODDataset filters.
        #
        # The item CATALOG (title/genres, used for item_text) is deliberately
        # taken from the UNFILTERED frame. Title and genres are side information,
        # not labels, so excluding an item's caption leaks nothing; and the
        # filter hits train hardest (early rows have the shortest histories), so
        # deriving the catalog from the filtered frame collapsed the train
        # item_text block to ~501 items against ~2210 for val. With Q-Former at
        # ~30M params that is memorised within 2 epochs: measured train g_itc
        # +4.82 against val -0.23 at the same n. Items dropped from the rows are
        # still needed here — they keep appearing inside other rows' history, so
        # Stage 3 does encode them and they do need an alignment.
        rows_before_history_filter = len(df)
        catalog_df = df
        if int(min_positive_history) > 1:
            df = df[df["his"].map(len) >= int(min_positive_history)].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"min_positive_history={min_positive_history} filtered out every row "
                    f"of {input_pkl_path} ({rows_before_history_filter} rows before)."
                )

        pos_df = df[df["label"] == 1].copy()
        if pos_df.empty:
            raise ValueError("No positive samples found.")

        def parse_genres(genres) -> list[str]:
            return sorted({g.strip() for g in str(genres).split("|") if g.strip()})

        # Step 1: build item metadata lookup tables.
        #
        # Example:
        #   item_titles[10] = "Toy Story (1995)"
        #   item_genres[10] = ["Animation", "Children's", "Comedy"]
        #
        # These lookups are reused by item-text and user-item samples.
        #
        # Built from ``catalog_df`` (pre-history-filter) rather than ``df``: see
        # the filter comment above. This is a superset of the items appearing in
        # ``df``, so every lookup made from pos_df below is still satisfied.
        item_titles: dict[int, str] = {}
        item_genres: dict[int, list[str]] = {}
        for iid, title, genres in catalog_df[["iid", "title", "genres"]].drop_duplicates("iid").itertuples(index=False):
            parsed_genres = parse_genres(genres)
            if not parsed_genres:
                raise ValueError(f"Item {iid} has empty genres.")
            item_titles[int(iid)] = str(title)
            item_genres[int(iid)] = parsed_genres

        def format_item_text(iid: int) -> str:
            return f"Title: {item_titles[int(iid)]}. Genres: {', '.join(item_genres[int(iid)])}."

        samples = []

        # Step 2: item-text samples.
        #
        # Purpose:
        #   Align each item CF embedding with its natural-language metadata.
        #
        # Training use:
        #   i_left -> Q-Former item encoder
        #   text   -> frozen text encoder
        #   loss   -> symmetric item-text contrastive loss
        #
        # Example output:
        #   {
        #       "sample_type": "item_text",
        #       "u": 0,
        #       "i_left": 10,
        #       "i_right": 0,
        #       "text": "Title: Toy Story (1995). Genres: Animation, Children's, Comedy.",
        #       "instruction": "...",
        #       "weight": 1.0,
        #   }
        use_item_holdout = (item_text_valid_frac + item_text_test_frac) > 0.0
        num_item_text = 0
        for iid in sorted(item_titles):
            if use_item_holdout:
                bucket = QFormerAlignmentBuilder._item_text_bucket(
                    int(iid), item_text_split_seed, item_text_valid_frac, item_text_test_frac
                )
                if bucket != item_text_split:
                    continue
            samples.append(
                {
                    "sample_type": "item_text",
                    "u": 0,
                    "i_left": int(iid),
                    "i_right": 0,
                    "text": format_item_text(int(iid)),
                    "instruction": rng.choice(QFormerAlignmentBuilder.TEMPL_ITEM_TEXT),
                    "weight": 1.0,
                }
            )
            num_item_text += 1

        # Step 3: item-item samples.
        #
        # Purpose:
        #   Preserve collaborative filtering/co-watch signal in the Q-Former
        #   space. This is the low-risk ILM-style addition beyond item-text.
        #
        # Transform:
        #   For each user, sort positive interactions by timestamp. For every
        #   item, pair it with the next item_pair_window positive items.
        #
        # Example with item_pair_window = 2:
        #   positive sequence: [10, 33, 41, 52]
        #   pairs: (10,33), (10,41), (33,41), (33,52), (41,52)
        #
        # Duplicate pairs across users are counted. The count is stored as
        # weight. The current trainer uses in-batch negatives and does not
        # apply weight yet, but the value is useful for later weighted sampling.
        #
        # Training use:
        #   i_left, i_right -> two Q-Former item encodings
        #   loss            -> symmetric pair contrastive loss
        item_pair_counts: dict[tuple[int, int], int] = {}
        sorted_pos = pos_df.sort_values(["uid", "timestamp"])
        window = max(int(item_pair_window), 1)
        for _, group in sorted_pos.groupby("uid"):
            pos_items = [int(iid) for iid in group["iid"].tolist() if int(iid) != 0]
            for left_idx, left_item in enumerate(pos_items):
                right_stop = min(left_idx + window + 1, len(pos_items))
                for right_item in pos_items[left_idx + 1 : right_stop]:
                    if left_item == right_item:
                        continue
                    pair = tuple(sorted((left_item, int(right_item))))
                    item_pair_counts[pair] = item_pair_counts.get(pair, 0) + 1

        item_item_samples = [
            {
                "sample_type": "item_item",
                "u": 0,
                "i_left": int(i1),
                "i_right": int(i2),
                "text": "",
                "instruction": rng.choice(QFormerAlignmentBuilder.TEMPL_ITEM_ITEM),
                "weight": float(weight),
            }
            for (i1, i2), weight in item_pair_counts.items()
        ]
        item_item_samples.sort(key=lambda x: (-x["weight"], x["i_left"], x["i_right"]))
        if max_item_item_pairs is not None:
            item_item_samples = item_item_samples[: int(max_item_item_pairs)]
        samples.extend(item_item_samples)

        # Step 4: optional user-item samples.
        #
        # ILM representation learning is item-centric. Stage 2 in this repo also
        # injects item/history tokens rather than a user CF token, so the default
        # is to leave these samples out of stage 1.
        #
        # Transform:
        #   Each positive row becomes one (user, positive item) pair.
        #
        # Training use:
        #   u      -> Q-Former user encoder
        #   i_left -> Q-Former item encoder
        #   loss   -> symmetric user-item contrastive loss with in-batch negatives
        #
        # Example output:
        #   {
        #       "sample_type": "user_item",
        #       "u": 1,
        #       "i_left": 33,
        #       "i_right": 0,
        #       "text": "Title: Grumpier Old Men (1995). Genres: Comedy, Romance.",
        #       "instruction": "...",
        #       "weight": 1.0,
        #   }
        user_item_samples = []
        if include_user_item:
            user_item_samples = [
                {
                    "sample_type": "user_item",
                    "u": int(row.uid),
                    "i_left": int(row.iid),
                    "i_right": 0,
                    "text": format_item_text(int(row.iid)),
                    "instruction": rng.choice(QFormerAlignmentBuilder.TEMPL_USER_ITEM),
                    "weight": 1.0,
                    # History BEFORE this interaction (padding id 0 stripped,
                    # capped to the max_history_length most recent — the SAME
                    # cap Stage 3's MovieOODDataset applies, so the pooling is
                    # pretrained on the length regime it will actually run).
                    # Lets stage 1 represent the user by POOLING history items
                    # through cross-attention — the exact multi-source path
                    # Stage 3's <UserProfile> uses — instead of the single MF
                    # user vector (S=1), which left the pooling behaviour
                    # untrained until Stage 3.
                    "his": [int(x) for x in row.his if int(x) != 0][-int(max_history_length):],
                }
                for row in pos_df.itertuples(index=False)
            ]
            if max_user_item_pairs is not None:
                user_item_samples = user_item_samples[: int(max_user_item_pairs)]
            samples.extend(user_item_samples)

        # Step 5: persist one flat list.
        #
        # The stage-1 trainer uses a normal DataLoader over this list, then
        # groups rows by sample_type inside train_step. Keeping one flat list
        # makes the dataset easy to inspect and avoids hidden banks/legacy views.
        stats = {
            "num_samples": len(samples),
            "num_item_text": num_item_text,
            "num_item_item": len(item_item_samples),
            "num_user_item": len(user_item_samples),
            "num_users": int(pos_df["uid"].nunique()),
            "num_items": len(item_titles),
            "num_positive_rows": int(len(pos_df)),
            "item_pair_window": window,
            "include_user_item": bool(include_user_item),
            "max_history_length": int(max_history_length),
            "item_text_split": item_text_split,
            "item_text_valid_frac": float(item_text_valid_frac),
            "item_text_test_frac": float(item_text_test_frac),
            "min_positive_history": int(min_positive_history),
            "rows_before_history_filter": int(rows_before_history_filter),
            "rows_after_history_filter": int(len(df)),
            # Real history length of the user_item samples, padding excluded.
            # The min_positive_history filter counts len(his) INCLUDING the
            # padding id 0, so his=[0, 10] passes a threshold of 2 while giving
            # the Q-Former a single cross-attention key — and pooling over one
            # key is a no-op. Record the distribution so a mostly-degenerate
            # user_item block is visible here instead of having to be inferred
            # from Stage 3 behaviour.
            "user_item_his_len_mean": (
                sum(len(s["his"]) for s in user_item_samples) / len(user_item_samples)
                if user_item_samples
                else 0.0
            ),
            "user_item_his_len_lt2_frac": (
                sum(1 for s in user_item_samples if len(s["his"]) < 2)
                / len(user_item_samples)
                if user_item_samples
                else 0.0
            ),
            # Catalog comes from the unfiltered frame; the rows do not. A large
            # gap here is expected and is the point — it is the number of items
            # that keep appearing inside other rows' history and therefore still
            # need an item_text alignment.
            "catalog_items_unfiltered": int(catalog_df["iid"].nunique()),
            "catalog_items_after_filter": int(df["iid"].nunique()),
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(
            {
                "seed": seed,
                "samples": samples,
                "stats": stats,
            },
            output_path,
        )
        return output_path, len(samples)
