"""
=============================================================================
CASTLE-Pegon -- Phase 08: ByT5-base Fine-tuning (Tier 5)
=============================================================================
Tujuan:
    Fine-tune ByT5-base untuk post-OCR correction sebagai Tier 5 baseline.
    Dijalankan DUA KALI untuk dua kondisi:

    Tier 5a: ByT5 + random augmentation
        python src/phase08_byt5.py --mode random

    Tier 5b: ByT5 + glyph augmentation (metode utama)
        python src/phase08_byt5.py --mode glyph
        
    cd /root/autodl-tmp/castle-pegon
    nohup python src/phase08_byt5.py --mode glyph \
        > /root/castle-results/tier5_glyph/training.log 2>&1 &
        echo "PID: $!"
        tail -f /root/castle-results/tier5_glyph/training.log
    nohup python src/phase08_byt5.py --mode random \
        > /root/castle-results/tier5_random/training.log 2>&1 &
        echo "PID: $!"
        tail -f /root/castle-results/tier5_random/training.log

Desain:
    Train input  : synthetic noisy text (random atau glyph)
    Train target : ground truth (clean)
    Val input    : GPT-4o OCR output (real OCR)
    Val target   : ground truth (clean)
    Test input   : GPT-4o OCR output (real OCR)
    Test target  : ground truth (clean)

Input:
    data/augmented/train_glyph.json  atau  train_random.json
    data/augmented/val.json
    data/augmented/test.json
    models/byt5-base/

Output:
    results/tier5_{mode}/
        checkpoint-best/     ← best model by val CER
        predictions_test.json
        metrics.json

Run:
    python src/phase08_byt5.py --mode glyph   (Tier 5b)
    python src/phase08_byt5.py --mode random  (Tier 5a)
=============================================================================
"""

import os
import sys
import json
import argparse
import unicodedata
import numpy as np
import torch
from pathlib import Path

from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR, PROCESSED_DIR

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

AUG_DIR    = os.path.join(BASE_DIR, "data", "augmented")
MODEL_PATH = os.path.join(BASE_DIR, "models", "byt5-base")

MAX_INPUT_LENGTH  = 256
MAX_TARGET_LENGTH = 256
BATCH_SIZE        = 8
GRAD_ACCUM        = 8      # effective batch = 64
LEARNING_RATE     = 3e-4
NUM_EPOCHS        = 20
WARMUP_RATIO      = 0.1
WEIGHT_DECAY      = 0.01
PATIENCE          = 5
LABEL_SMOOTHING   = 0.1
NUM_BEAMS         = 4
MAX_GEN_LENGTH    = 256
SEED              = 42

torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISASI
# ─────────────────────────────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\u0640", "")
    return " ".join(text.split()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class PegonDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, split: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self.data = [
            d for d in raw
            if d.get('input', '').strip()
            and d.get('target', '').strip()
        ]
        self.tokenizer = tokenizer

        # Distribution info
        from collections import Counter
        aug_types = Counter(d.get('aug_type', 'unknown') for d in self.data)
        print(f"  [{split}] {len(self.data)} pairs")
        print(f"    Types: {dict(aug_types)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        inp    = item['input'].strip()
        target = item['target'].strip()

        model_input = self.tokenizer(
            inp,
            max_length     = MAX_INPUT_LENGTH,
            padding        = False,
            truncation     = True,
            return_tensors = None,
        )
        labels = self.tokenizer(
            text_target    = target,
            max_length     = MAX_TARGET_LENGTH,
            padding        = False,
            truncation     = True,
            return_tensors = None,
        )
        model_input['labels'] = labels['input_ids']
        return model_input


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] \
                    else 1 + min(prev, dp[j], dp[j-1])
            prev  = temp
    return dp[n]


def compute_metrics_fn(tokenizer):
    vocab_size = tokenizer.vocab_size

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        predictions  = np.clip(predictions, 0, vocab_size - 1).astype(np.int32)
        labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels_clean = np.clip(labels_clean, 0, vocab_size - 1).astype(np.int32)

        decoded_preds  = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            labels_clean, skip_special_tokens=True
        )

        decoded_preds  = [normalize_arabic(p.strip()) for p in decoded_preds]
        decoded_labels = [normalize_arabic(l.strip()) for l in decoded_labels]

        valid = [
            (p, r) for p, r in zip(decoded_preds, decoded_labels)
            if r and p
        ]
        if not valid:
            return {'cer': 1.0, 'wer': 1.0}

        preds, refs = zip(*valid)

        cer_scores = [
            edit_distance(p, r) / len(r)
            for p, r in zip(preds, refs)
        ]
        wer_scores = [
            edit_distance(p.split(), r.split()) / max(len(r.split()), 1)
            for p, r in zip(preds, refs)
        ]

        return {
            'cer': round(float(np.mean(cer_scores)), 4),
            'wer': round(float(np.mean(wer_scores)), 4),
        }

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE & SAVE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_save(model, tokenizer, data_path: str,
                       output_path: str, split_name: str,
                       batch_size: int = 16):
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    model.eval()

    with open(data_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    data = [
        d for d in raw
        if d.get('input', '').strip() and d.get('target', '').strip()
    ]

    results   = []
    all_preds = []
    all_refs  = []

    print(f"\nGenerating on {split_name} ({len(data)} records)...")

    for i in range(0, len(data), batch_size):
        batch       = data[i:i + batch_size]
        inputs_text = [d['input'].strip() for d in batch]

        enc = tokenizer(
            inputs_text,
            max_length     = MAX_INPUT_LENGTH,
            padding        = True,
            truncation     = True,
            return_tensors = 'pt',
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens = MAX_GEN_LENGTH,
                num_beams      = NUM_BEAMS,
                early_stopping = True,
            )

        preds = tokenizer.batch_decode(out, skip_special_tokens=True)

        for d, pred in zip(batch, preds):
            ref  = normalize_arabic(d['target'].strip())
            pred = normalize_arabic(pred.strip())
            all_preds.append(pred)
            all_refs.append(ref)
            results.append({
                'id'        : d.get('id', ''),
                'source_id' : d.get('source_id', ''),
                'input'     : d['input'],
                'prediction': pred,
                'target'    : ref,
                'kitab_id'  : d.get('kitab_id', ''),
            })

        if (i // batch_size + 1) % 10 == 0:
            print(f"  {min(i+batch_size, len(data))}/{len(data)} done...")

    # Compute metrics
    valid = [(p, r) for p, r in zip(all_preds, all_refs) if r and p]
    if valid:
        preds_v, refs_v = zip(*valid)
        cer = round(float(np.mean([
            edit_distance(p, r) / len(r)
            for p, r in zip(preds_v, refs_v)
        ])), 4)
        wer = round(float(np.mean([
            edit_distance(p.split(), r.split()) / max(len(r.split()), 1)
            for p, r in zip(preds_v, refs_v)
        ])), 4)
        n_empty = sum(1 for p in all_preds if not p)
        coverage = round((len(valid) - n_empty) / len(valid) * 100, 1)
    else:
        cer = wer = 1.0
        coverage = 0.0

    metrics = {
        'split'       : split_name,
        'n'           : len(valid),
        'CER'         : cer,
        'WER'         : wer,
        'coverage_pct': coverage,
    }

    print(f"\n  {split_name} Results:")
    print(f"    CER      = {cer:.4f}")
    print(f"    WER      = {wer:.4f}")
    print(f"    Coverage = {coverage:.1f}%")

    # Sample predictions
    print(f"\n  Sample predictions:")
    for r in results[:3]:
        print(f"    Input : {r['input'][:45]}")
        print(f"    Target: {r['target'][:45]}")
        print(f"    Pred  : {r['prediction'][:45]}")
        print()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'predictions': results},
                  f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["glyph", "random"],
        default="glyph",
        help="Training data: glyph (Tier 5b) or random (Tier 5a)"
    )
    return parser.parse_args()


def main():
    args    = parse_args()
    mode    = args.mode
    tier    = "5b" if mode == "glyph" else "5a"

    OUTPUT_DIR   = f"/root/castle-results/tier5_{mode}"
    BEST_CKPT    = os.path.join(OUTPUT_DIR, "checkpoint-best")
    TRAIN_FILE   = os.path.join(AUG_DIR, f"train_{mode}_nooriginal.json")
    VAL_FILE     = os.path.join(AUG_DIR, "val.json")
    TEST_FILE    = os.path.join(AUG_DIR, "test.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"CASTLE-Pegon -- Phase 08: ByT5 Tier {tier}")
    print("=" * 60)
    print(f"  Mode   : {mode} augmentation")
    print(f"  Train  : {TRAIN_FILE}")
    print(f"  Val    : {VAL_FILE}")
    print(f"  Test   : {TEST_FILE}")
    print(f"  Output : {OUTPUT_DIR}")

    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # Load tokenizer & model
    print(f"\nLoading ByT5-base from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    params    = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.0f}M")

    # Datasets
    print("\nLoading datasets...")
    train_ds = PegonDataset(TRAIN_FILE, tokenizer, 'train')
    val_ds   = PegonDataset(VAL_FILE,   tokenizer, 'val')

    # Data collator
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model              = model,
        label_pad_token_id = -100,
        pad_to_multiple_of = 8,
    )

    # Training args
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    print(f"\n  Precision : {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    print(f"  Batch     : {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
    print(f"  Epochs    : {NUM_EPOCHS} (patience={PATIENCE})")

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        label_smoothing_factor      = LABEL_SMOOTHING,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        predict_with_generate       = True,
        generation_num_beams        = NUM_BEAMS,
        generation_max_length       = MAX_GEN_LENGTH,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,
        logging_steps               = 20,
        save_total_limit            = 2,
        save_only_model             = True,
        report_to                   = "none",
        seed                        = SEED,
        dataloader_num_workers      = 2,
    )

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
        data_collator    = collator,
        compute_metrics  = compute_metrics_fn(tokenizer),
        callbacks        = [
            EarlyStoppingCallback(early_stopping_patience=PATIENCE)
        ],
    )

    # Train
    print(f"\nStarting training (Tier {tier} — {mode} augmentation)...")
    trainer.train()

    # Save best model
    print(f"\nSaving best model to {BEST_CKPT}...")
    trainer.save_model(BEST_CKPT)
    tokenizer.save_pretrained(BEST_CKPT)

    # Evaluate on val
    val_metrics = evaluate_and_save(
        model, tokenizer,
        VAL_FILE,
        os.path.join(OUTPUT_DIR, 'predictions_val.json'),
        'val'
    )

    # Evaluate on test
    test_metrics = evaluate_and_save(
        model, tokenizer,
        TEST_FILE,
        os.path.join(OUTPUT_DIR, 'predictions_test.json'),
        'test'
    )

    # Save final metrics
    final_metrics = {
        'tier'    : tier,
        'mode'    : mode,
        'model'   : 'ByT5-base',
        'val'     : val_metrics,
        'test'    : test_metrics,
        'config'  : {
            'train_file'   : TRAIN_FILE,
            'batch_size'   : BATCH_SIZE,
            'grad_accum'   : GRAD_ACCUM,
            'lr'           : LEARNING_RATE,
            'epochs'       : NUM_EPOCHS,
            'patience'     : PATIENCE,
            'num_beams'    : NUM_BEAMS,
            'label_smooth' : LABEL_SMOOTHING,
        }
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TIER {tier} COMPLETE — ByT5 ({mode} augmentation)")
    print(f"{'='*60}")
    print(f"  Val  CER : {val_metrics['CER']:.4f}")
    print(f"  Test CER : {test_metrics['CER']:.4f}")
    print(f"  Test WER : {test_metrics['WER']:.4f}")
    print(f"  Checkpoint: {BEST_CKPT}")

    if mode == "glyph":
        print(f"\n  Tier 5a (random) belum dijalankan:")
        print(f"  python src/phase08_byt5.py --mode random")

    print(f"\n  NEXT: python src/phase09_castle.py")
    print("=" * 60)


if __name__ == "__main__":
    main()