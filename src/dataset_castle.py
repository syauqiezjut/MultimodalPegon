"""
CASTLE-Pegon Dataset
Loads: line image + GPT-4o OCR text → ground truth
"""

import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import unicodedata
import re


class PegonCorrectionDataset(Dataset):
    """Dataset for multimodal post-OCR correction."""

    def __init__(
        self,
        json_path: str,
        image_root: str,
        vit_processor,
        byt5_tokenizer,
        max_source_length: int = 256,
        max_target_length: int = 256,
        split: str = "train",
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.vit_processor = vit_processor
        self.byt5_tokenizer = byt5_tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.split = split

        # Verify images exist
        missing = 0
        for rec in self.data:
            img_path = self._get_image_path(rec)
            if not os.path.exists(img_path):
                missing += 1
        if missing > 0:
            print(f"WARNING: {missing}/{len(self.data)} images missing in {split}")

    def _get_image_path(self, record):
        """Convert Mac path to AutoDL path."""
        filename = os.path.basename(record["cropped_image_path"])
        kitab_id = record["kitab_id"]
        return os.path.join(self.image_root, kitab_id, filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]

        # --- Image ---
        img_path = self._get_image_path(rec)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback: white image if corrupted
            print(f"WARNING: Failed to load {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        # Pad to square (white background) then let processor resize to 224x224
        w, h = image.size
        max_side = max(w, h)
        padded = Image.new("RGB", (max_side, max_side), (255, 255, 255))
        padded.paste(image, (0, (max_side - h) // 2))  # Center vertically
        pixel_values = self.vit_processor(
            images=padded, return_tensors="pt"
        ).pixel_values.squeeze(0)  # [3, 224, 224]

        # --- Source text (OCR input) ---
        ocr_text = rec.get("ocr_gpt4o", "")
        if not ocr_text:
            ocr_text = ""
        source_encoding = self.byt5_tokenizer(
            ocr_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = source_encoding.input_ids.squeeze(0)
        attention_mask = source_encoding.attention_mask.squeeze(0)

        # --- Target text (ground truth) ---
        gt_text = rec.get("ground_truth", "")
        target_encoding = self.byt5_tokenizer(
            gt_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids.squeeze(0)
        # Replace padding token id with -100 for loss computation
        labels[labels == self.byt5_tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,       # [3, 224, 224]
            "input_ids": input_ids,             # [max_source_length]
            "attention_mask": attention_mask,    # [max_source_length]
            "labels": labels,                   # [max_target_length]
            "record_id": rec["record_id"],
        }


def normalize_pegon(text: str) -> str:
    """Unicode normalization for CER computation.
    NFC + remove ZWNJ/ZWJ/kashida.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove ZWNJ (U+200C), ZWJ (U+200D), kashida/tatweel (U+0640)
    text = re.sub(r"[\u200c\u200d\u0640]", "", text)
    text = text.strip()
    return text