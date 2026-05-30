"""
TrOCR Fine-Tuning on BD Medicine Prescription Data
====================================================
Adapts microsoft/trocr-large-handwritten to BD medicine vocabulary using
synthetic prescription images generated from the medicine database.

This script is designed to run on Google Colab (free T4 GPU, ~2-3 hours).

HOW TO USE
----------
1. Run generate_synthetic_data.py locally to produce data/synthetic_train/
2. Upload synthetic_train/ to Google Drive
3. Open Google Colab, mount Drive, and run this script
4. Download the fine-tuned model and point _MODEL_ID in ocr_engine.py to it

OR run directly with --local if you have a GPU locally.

COLAB SETUP (paste in first cell)
----------------------------------
!pip install transformers datasets pillow torch torchvision tqdm
from google.colab import drive; drive.mount('/content/drive')
import subprocess; subprocess.run(['python3', '/content/drive/MyDrive/finetune_trocr.py',
    '--data-dir', '/content/drive/MyDrive/synthetic_train',
    '--output-dir', '/content/drive/MyDrive/trocr-bd-medicines'])

LOCAL GPU (if available)
------------------------
python3 scripts/finetune_trocr.py \\
    --data-dir data/synthetic_train \\
    --output-dir models/trocr-bd-medicines \\
    --epochs 3 --batch-size 8

Expected results after fine-tuning on 5,000 synthetic BD medicine images:
  • CER on BD medicine names: expected improvement from ~5% → ~1.5%
  • Brand name exact match: expected improvement from ~85% → ~97%
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

log = logging.getLogger("finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

BASE_DIR = Path(__file__).parent.parent if "__file__" in dir() else Path(".")


# ── Dataset ───────────────────────────────────────────────────────────────────

class PrescriptionDataset:
    """
    HuggingFace-compatible dataset for TrOCR fine-tuning.
    Expects data_dir/ to contain:
      images/   — PNG image files
      labels.csv — columns: file_name, text
    """

    def __init__(self, data_dir: Path, processor, augment: bool = True):
        self.data_dir  = data_dir
        self.processor = processor
        self.augment   = augment
        self.items: list[dict] = []
        self._load()

    def _load(self):
        labels_path = self.data_dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv not found in {self.data_dir}")
        with open(labels_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_path = self.data_dir / row["file_name"]
                if img_path.exists():
                    self.items.append({"image_path": img_path, "text": row["text"]})
        log.info("Dataset: %d items from %s", len(self.items), self.data_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image, ImageFilter
        import numpy as np

        item = self.items[idx]
        img  = Image.open(item["image_path"]).convert("RGB")

        if self.augment:
            # Additional online augmentation (each epoch sees different variants)
            arr = np.array(img, dtype=np.float32)
            noise = np.random.randn(*arr.shape) * random.uniform(0, 12)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            item["text"],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id's of the labels with -100 so it's ignored in the loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    import torch
    from torch.utils.data import DataLoader, random_split
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        default_data_collator,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device == "cpu":
        log.warning("Running on CPU — fine-tuning will be very slow. Use Colab T4 GPU.")

    model_id   = args.base_model
    output_dir = Path(args.output_dir)
    data_dir   = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading base model: %s", model_id)
    processor = TrOCRProcessor.from_pretrained(model_id)
    model     = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Required config for seq2seq
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size

    # Build dataset
    dataset = PrescriptionDataset(data_dir, processor, augment=True)
    n_val   = min(int(len(dataset) * 0.1), 500)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    log.info("Train: %d  Val: %d", n_train, n_val)

    # Metric: CER
    def compute_metrics(pred):
        from rapidfuzz.distance import Levenshtein
        labels_ids = pred.label_ids
        pred_ids   = pred.predictions

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_ids = labels_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer_list = [
            Levenshtein.normalized_distance(p.lower(), l.lower())
            for p, l in zip(pred_str, label_str) if l
        ]
        return {"cer": sum(cer_list) / len(cer_list) if cer_list else 0.0}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=5e-5,
        warmup_steps=100,
        fp16=(device == "cuda"),
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=(device == "cuda"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    log.info("Starting fine-tuning…")
    trainer.train()

    # Save
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    log.info("Model saved to %s", output_dir)

    # Evaluate on val
    metrics = trainer.evaluate()
    log.info("Final eval CER: %.4f", metrics.get("eval_cer", -1))

    # Save metadata
    with open(output_dir / "finetune_metadata.json", "w") as f:
        json.dump({
            "base_model": model_id,
            "epochs": args.epochs,
            "n_train": n_train,
            "n_val": n_val,
            "final_cer": metrics.get("eval_cer"),
        }, f, indent=2)

    print()
    print("=" * 60)
    print("Fine-tuning complete!")
    print(f"  Model saved to:  {output_dir}")
    print(f"  Final CER:       {metrics.get('eval_cer', '?'):.4f}")
    print()
    print("To use this model in the app, set in ocr_engine.py:")
    print(f'  _MODEL_ID = "{output_dir.resolve()}"')
    print("=" * 60)


# ── Quick test after fine-tuning ───────────────────────────────────────────────

def test_model(model_dir: str, test_image: str):
    """Quick sanity check: run the fine-tuned model on a single image."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image

    processor = TrOCRProcessor.from_pretrained(model_dir)
    model     = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.eval()

    img = Image.open(test_image).convert("RGB")
    pv  = processor(images=img, return_tensors="pt").pixel_values

    with torch.no_grad():
        ids = model.generate(pv, num_beams=4, max_new_tokens=64)

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"Predicted: {text}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fine-tune TrOCR on BD medicine prescription data")
    ap.add_argument("--data-dir",    required=True, help="Path to synthetic_train/ directory")
    ap.add_argument("--output-dir",  required=True, help="Where to save the fine-tuned model")
    ap.add_argument("--base-model",  default="microsoft/trocr-large-handwritten",
                    help="Base TrOCR model to fine-tune")
    ap.add_argument("--epochs",      type=int, default=5)
    ap.add_argument("--batch-size",  type=int, default=4,
                    help="Batch size (use 8-16 on T4 GPU, 4 on small GPU, 2 on CPU)")
    ap.add_argument("--test-image",  help="Quick test: run fine-tuned model on this image")
    args = ap.parse_args()

    if args.test_image:
        test_model(args.output_dir, args.test_image)
    else:
        train(args)


if __name__ == "__main__":
    main()
