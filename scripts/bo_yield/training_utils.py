"""Training utilities for fine-tuning yield prediction models."""

import torch
import torch.nn.functional as F
from transformers import Trainer, DataCollatorWithPadding
from typing import Dict, Any


class CollatorForYield:
    """Data collator for yield prediction."""

    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        has_labels = "labels" in features[0]
        if has_labels:
            labels = torch.tensor([float(f["labels"]) for f in features], dtype=torch.float)
        token_feats = [{k: v for k, v in f.items() if k in ("input_ids", "attention_mask")} for f in features]
        batch = self.pad(token_feats)
        if has_labels:
            batch["labels"] = labels
        return batch


class YieldTrainer(Trainer):
    """Trainer for yield prediction with MSE loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        preds = model(**model_inputs).squeeze(-1)
        if labels is None:
            loss = preds.new_zeros(())
        else:
            loss = F.mse_loss(preds, labels)
        return (loss, preds) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("labels")

        with torch.no_grad():
            loss, preds = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, preds, labels)


class YieldDataset(torch.utils.data.Dataset):
    """Dataset for yield prediction."""

    def __init__(self, texts, y, tokenizer, max_length=512):
        self.enc = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.enc["input_ids"][i], dtype=torch.long),
            "attention_mask": torch.tensor(self.enc["attention_mask"][i], dtype=torch.long),
            "labels": torch.tensor(self.y[i], dtype=torch.float),  # [%]
        }
