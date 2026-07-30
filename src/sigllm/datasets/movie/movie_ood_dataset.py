"""Placeholder concrete dataset for MovieDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import numpy as np

from sigllm.common.config import Config
from sigllm.common.logging_utils import NotebookLogger
from sigllm.datasets.base.rec_base_dataset import RecBaseDataset

LOGGER = NotebookLogger.rich_logger("sigllm.movie_ood_dataset")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

class MovieOODDataset(RecBaseDataset):

	def __init__(
		self,
		dataset_config,
		filename: str = None,
		subset: Literal["all", "warm", "cold"] = "all"
	) -> None:
		ann_path = Path(dataset_config.build_info.storage) / filename
		
		if (ann_path is None) or (not ann_path.exists()):
			raise ValueError(f"Annotation path {ann_path} does not exist.")
		
		df = pd.read_pickle(ann_path.with_suffix(".pkl")).reset_index(drop=True)

		min_positive_history = 2

		if hasattr(dataset_config, "get"):
			try:
				min_positive_history = int(dataset_config.get("min_positive_history", 2))
			except Exception:
				min_positive_history = 2

		if min_positive_history > 1 and "his" in df.columns:
			before = len(df)
			df = df[df["his"].map(len) >= min_positive_history].reset_index(drop=True)
			log_step(
				"SeLLa-parity history filter",
				f"kept {len(df) / before:.4f} rows (min_positive_history={min_positive_history})"
			)

		# The filter above counts len(his), which INCLUDES the padding id 0, so
		# his=[0, 10] passes min_positive_history=2 while carrying only ONE real
		# item. That matters a lot downstream: <UserProfile> pools the history
		# through Q-Former cross-attention, and pooling over a single key is a
		# no-op — all the multi-source machinery Stage 1 pretrained (padded
		# sequences, source masks, candidate-conditioned selection) does nothing
		# for those rows. Report the REAL distribution so the gap between
		# "passed the filter" and "has usable history" is visible rather than
		# inferred from one prompt-preview line.
		if "his" in df.columns and len(df):
			real_len = df["his"].map(lambda h: sum(1 for x in h if int(x) != 0))
			total = len(real_len)
			log_step(
				"History length (real items, padding excluded)",
				f"mean={real_len.mean():.2f} median={int(real_len.median())} "
				f"p10={int(real_len.quantile(0.10))} p90={int(real_len.quantile(0.90))} "
				f"max={int(real_len.max())} | "
				f"<1: {(real_len < 1).sum() / total:.1%}, "
				f"<2: {(real_len < 2).sum() / total:.1%}, "
				f"<5: {(real_len < 5).sum() / total:.1%} "
				f"(a row with <2 real items makes <UserProfile> pooling a no-op)",
			)

		self.annotation = df.copy()

		warm_definition = "not_cold"

		if hasattr(dataset_config, "get"):
			try:
				warm_definition = str(dataset_config.get("warm_definition", "not_cold"))
			except Exception:
				warm_definition = "not_cold"

		if subset == "warm":
			if warm_definition == "threshold" and "warm" in df.columns:
				self.annotation = df[df['warm'].isin([1])].copy()
			else:
				self.annotation = df[df['not_cold'].isin([1])].copy()

		if subset == "cold":
			self.annotation = df[df['not_cold'].isin([0])].copy()

		self.use_his = False
		self.prompt_flag = False

		if "sessionItems" in self.annotation.columns or "his" in self.annotation.columns:
			used_columns = ['uid','iid','title','his', 'his_title','label']
			renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']

			if 'not_cold' in self.annotation.columns:
				used_columns.append('not_cold')
				renamed_columns.append('prompt_flag')
				self.prompt_flag = True
			
			self.use_his = True
			self.annotation = self.annotation[used_columns]
			self.annotation.columns = renamed_columns
			
			self.annotation['InteractedItemIDs'] = self.annotation['InteractedItemIDs'].map(list)
			self.annotation['InteractedItemTitles'] = self.annotation['InteractedItemTitles'].map(list)
		else:
			used_columns = ['uid','iid','title','label']
			renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
			if 'not_cold' in self.annotation.columns:
				used_columns.append('not_cold')
				renamed_columns.append('prompt_flag')
				self.prompt_flag = True
			
			self.annotation = self.annotation[used_columns]
			self.annotation.columns = renamed_columns
		
		log_step("data path", f"{ann_path} | data size: {self.annotation.shape}")
		self.user_num = self.annotation['UserID'].max() + 1
		self.item_num = self.annotation['TargetItemID'].max() + 1

		if self.use_his:
			# SHARED history cap (datasets.*.max_history_length): the same key
			# drives the Q-Former alignment builder's `his` truncation, so the
			# Stage-1 history pooling is pretrained on exactly the sequence
			# length this dataset feeds Stage 3. Keep them on one key — a
			# hardcoded 10 here vs 50 in the builder trains the pooling on a
			# length regime it never sees again.
			history_cap = 10
			if hasattr(dataset_config, "get"):
				try:
					history_cap = int(dataset_config.get("max_history_length", 10))
				except Exception:
					history_cap = 10
			max_length = 0
			for his in self.annotation['InteractedItemIDs']:
				max_length = max(max_length, len(his))
			self.max_length = min(max_length, history_cap)
			log_step("Movie OOD datasets, max history length:", str(self.max_length))
	
	def __getitem__(self, index):
		
		row = self.annotation.iloc[index]

		def _add_prompt_flag(sample: dict) -> dict:
			if self.prompt_flag:
				sample["prompt_flag"] = row["prompt_flag"]
			return sample

		user_id = row["UserID"]
		target_item_id = row["TargetItemID"]
		target_title = row["TargetItemTitle"].strip(" ")
		label = row["label"]
		
		if self.use_his:
			history_item_ids = row["InteractedItemIDs"]
			history_titles = row["InteractedItemTitles"]
			history_len = len(history_item_ids)

			interacted_count = history_len - 1 if (history_len > 0 and history_item_ids[0] == 0) else history_len

			max_history_len = self.max_length  

			if history_len < max_history_len:
				pad_size = max_history_len - history_len
				padded_history_ids = ([0] * pad_size) + list(history_item_ids)
			elif history_len > max_history_len:
				padded_history_ids = list(history_item_ids[-max_history_len:])
				interacted_count = max_history_len
			else:
				padded_history_ids = list(history_item_ids)

			recent_titles = history_titles[-interacted_count:] if interacted_count > 0 else []
			processed_titles = self.convert_title_list(recent_titles)

			sample = {
				"UserID": user_id,
				"InteractedItemIDs_pad": np.array(padded_history_ids),
				"InteractedItemTitles": processed_titles,
				"TargetItemID": target_item_id,
				"TargetItemTitle": f"\"{target_title}\"",
				"InteractedNum": interacted_count,
				"label": row["label"],
			}
			return _add_prompt_flag(sample)
		else:
			sample = {
				"UserID": user_id,
				"TargetItemID": target_item_id,
				"TargetItemTitle": target_title,
				"label": label,
			}
			return _add_prompt_flag(sample)
		
