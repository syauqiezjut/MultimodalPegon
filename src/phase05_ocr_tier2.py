"""
=============================================================================
CASTLE-Pegon -- Phase 05: Tier 2 OCR Benchmarking
=============================================================================
Engine  : TrOCR variants + Arabic-Nougat + GOT-OCR2
Tujuan  : ViT-based encoder-decoder zero-shot pada Pegon
          Tunjukkan bahwa specialized OCR transformer masih struggle

Models:
    1. TrOCR-small-handwritten  (62M)   microsoft/trocr-small-handwritten
    2. TrOCR-base-handwritten   (334M)  microsoft/trocr-base-handwritten
    3. TrOCR-large-handwritten  (558M)  microsoft/trocr-large-handwritten
    4. Arabic-Nougat-large      (~250M) MohamedRashad/arabic-large-nougat
    5. GOT-OCR2                 (580M)  stepfun-ai/GOT-OCR2_0

Protocol:
    - Zero-shot inference (no fine-tuning)
    - Evaluasi pada test split (506 records)
    - Metrics: CER, WER, Coverage (RC)
    - Normalisasi Unicode sebelum metric computation

Input : data/processed/pegon_filtered.json
        data/splits/test.json
Output: results/ocr/tier2_metrics.json
        results/ocr/tier2_predictions_test.json

Run: python src/phase05_ocr_tier2.py
     python src/phase05_ocr_tier2.py --models trocr-small trocr-base
     python src/phase05_ocr_tier2.py --skip got-ocr2
=============================================================================
"""

import os
import sys
import json
import time
import argparse
import unicodedata
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, SPLITS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

FILTERED_JSON = os.path.join(PROCESSED_DIR, "pegon_filtered.json")
TEST_JSON     = os.path.join(SPLITS_DIR, "test.json")
RESULTS_DIR   = os.path.join(
    str(Path(__file__).parent.parent), "results", "ocr"
)

# Model registry
MODEL_REGISTRY = {
    "trocr-small": {
        "name"       : "TrOCR-small-handwritten",
        "hub_id"     : "models/trocr-small",
        "type"       : "trocr",
        "params"     : "62M",
        "description": "DeiT-Small encoder + MiniLM decoder",
    },
    "trocr-base": {
        "name"       : "TrOCR-base-handwritten",
        "hub_id"     : "models/trocr-base",
        "type"       : "trocr",
        "params"     : "334M",
        "description": "BEiT-Base encoder + RoBERTa-Large decoder",
    },
    "trocr-large": {
        "name"       : "TrOCR-large-handwritten",
        "hub_id"     : "models/trocr-large",
        "type"       : "trocr",
        "params"     : "558M",
        "description": "BEiT-Large encoder + RoBERTa-Large decoder",
    },
    "arabic-nougat": {
        "name"       : "Arabic-Nougat-large",
        "hub_id"     : "models/arabic-nougat",
        "type"       : "nougat",
        "params"     : "~250M",
        "description": "Nougat architecture fine-tuned on Arabic books",
    },
    "got-ocr2": {
        "name"       : "GOT-OCR2",
        "hub_id"     : "models/got-ocr2",
        "type"       : "got",
        "params"     : "580M",
        "description": "General OCR Theory v2, ViT encoder + Transformer decoder",
    },
}

REPORT_EVERY = 50

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────────────────────

import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ Device: CUDA ({torch.cuda.get_device_name(0)})")
    print(f"   VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print(f"✅ Device: Apple M1 MPS")
else:
    DEVICE = "cpu"
    print(f"⚠️  Device: CPU")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION
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
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_trocr(hub_id: str):
    """Load TrOCR model + processor."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    print(f"  Loading {hub_id}...")
    processor = TrOCRProcessor.from_pretrained(hub_id)
    model     = VisionEncoderDecoderModel.from_pretrained(hub_id)
    model     = model.to(DEVICE)
    model.eval()
    print(f"  ✅ Loaded")
    return processor, model


def load_nougat(hub_id: str):
    """Load Arabic-Nougat model + processor."""
    from transformers import NougatProcessor, VisionEncoderDecoderModel
    print(f"  Loading {hub_id}...")
    processor = NougatProcessor.from_pretrained(hub_id)
    model     = VisionEncoderDecoderModel.from_pretrained(hub_id)
    model     = model.to(DEVICE)
    model.eval()
    print(f"  ✅ Loaded")
    return processor, model


def load_got_ocr2(hub_id: str):
    """Load GOT-OCR2 model."""
    from transformers import AutoTokenizer, AutoModel
    print(f"  Loading {hub_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        hub_id, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        hub_id,
        trust_remote_code = True,
        low_cpu_mem_usage = True,
        device_map        = "auto" if DEVICE == "cuda" else None,
        use_safetensors   = True,
        pad_token_id      = tokenizer.eos_token_id,
    )
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    model = model.eval()
    print(f"  ✅ Loaded")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def infer_trocr(processor, model, image_path: str) -> str:
    """TrOCR inference pada satu line image."""
    from PIL import Image
    try:
        image  = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=256)
        text = processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]
        return text.strip()
    except Exception as e:
        return ""


def infer_nougat(processor, model, image_path: str) -> str:
    """Arabic-Nougat inference pada satu line image."""
    from PIL import Image
    try:
        image  = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                min_length    = 1,
                max_new_tokens= 256,
                bad_words_ids = [[processor.tokenizer.unk_token_id]],
            )
        text = processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]
        return text.strip()
    except Exception as e:
        return ""


def infer_got_ocr2(tokenizer, model, image_path: str) -> str:
    """GOT-OCR2 inference pada satu line image."""
    try:
        with torch.no_grad():
            result = model.chat(
                tokenizer,
                image_path,
                ocr_type = "ocr",
            )
        return str(result).strip()
    except Exception as e:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def edit_distance(s1, s2) -> int:
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


def compute_metrics(records: list, pred_key: str) -> dict:
    """
    Hitung CER/WER dengan tracking empty terpisah.
    pred_key: key di record dict untuk prediksi
    """
    cer_all, wer_all         = [], []
    cer_nonempty, wer_nonempty = [], []
    n_empty = 0

    for r in records:
        pred_raw = r.get(pred_key) or ""
        gt_raw   = r.get("ground_truth", "")
        if not gt_raw:
            continue

        pred = normalize_arabic(pred_raw)
        gt   = normalize_arabic(gt_raw)

        cer = edit_distance(pred, gt) / len(gt) if gt else 0.0
        wer = edit_distance(pred.split(), gt.split()) / \
              max(len(gt.split()), 1)

        cer_all.append(cer)
        wer_all.append(wer)

        if pred:
            cer_nonempty.append(cer)
            wer_nonempty.append(wer)
        else:
            n_empty += 1

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "n_total"         : len(cer_all),
        "n_empty_pred"    : n_empty,
        "n_nonempty_pred" : len(cer_nonempty),
        "coverage_pct"    : round(
            len(cer_nonempty) / max(len(cer_all), 1) * 100, 1
        ),
        "cer_incl_empty"  : avg(cer_all),
        "wer_incl_empty"  : avg(wer_all),
        "cer_excl_empty"  : avg(cer_nonempty),
        "wer_excl_empty"  : avg(wer_nonempty),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_key: str,
              model_cfg: dict,
              test_records: list,
              predictions: dict) -> dict:
    """
    Jalankan satu model pada semua test records.
    predictions: dict {record_id: {model_key: pred_text}}
    Resume-friendly: skip jika sudah ada.
    """
    print(f"\n{'─'*60}")
    print(f"Model: {model_cfg['name']} ({model_cfg['params']})")
    print(f"Hub  : {model_cfg['hub_id']}")
    print(f"{'─'*60}")

    # Cek berapa yang sudah selesai
    done = sum(
        1 for r in test_records
        if predictions.get(r['record_id'], {}).get(model_key) is not None
    )
    todo = len(test_records) - done
    print(f"  To process  : {todo}")
    print(f"  Already done: {done} (skip)")

    if todo == 0:
        print(f"  ✅ All records already processed")
        # Kembalikan hasil yang sudah ada
        return {
            r['record_id']: predictions.get(
                r['record_id'], {}
            ).get(model_key, "")
            for r in test_records
        }

    # Load model
    try:
        model_type = model_cfg['type']
        if model_type == "trocr":
            processor, model = load_trocr(model_cfg['hub_id'])
            infer_fn = lambda path: infer_trocr(processor, model, path)
        elif model_type == "nougat":
            processor, model = load_nougat(model_cfg['hub_id'])
            infer_fn = lambda path: infer_nougat(processor, model, path)
        elif model_type == "got":
            tokenizer, model = load_got_ocr2(model_cfg['hub_id'])
            infer_fn = lambda path: infer_got_ocr2(tokenizer, model, path)
        else:
            print(f"  ❌ Unknown model type: {model_type}")
            return {}
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        traceback.print_exc()
        return {}

    # Inference
    results  = {}
    t_start  = time.time()
    success  = 0
    failed   = 0

    for i, rec in enumerate(test_records):
        record_id = rec['record_id']

        # Skip jika sudah ada
        if predictions.get(record_id, {}).get(model_key) is not None:
            results[record_id] = predictions[record_id][model_key]
            continue

        img_path = rec.get('cropped_image_path', '')
        if not img_path or not os.path.isfile(img_path):
            results[record_id] = ""
            failed += 1
            continue

        try:
            text = infer_fn(img_path)
            results[record_id] = text
            success += 1
        except Exception as e:
            results[record_id] = ""
            failed += 1

        if (i + 1) % REPORT_EVERY == 0:
            elapsed   = time.time() - t_start
            per_img   = elapsed / max(success + failed, 1)
            remaining = (todo - success - failed) * per_img
            print(f"  [{i+1}/{len(test_records)}] "
                  f"ok={success} fail={failed} | "
                  f"{elapsed:.0f}s | ETA {remaining:.0f}s")

    elapsed = time.time() - t_start
    print(f"\n  ✅ Done: {success} ok, {failed} failed | {elapsed:.1f}s")
    if success > 0:
        print(f"  Avg: {elapsed/success*1000:.0f}ms per image")

    # Cleanup GPU memory
    if DEVICE == "cuda":
        del model
        torch.cuda.empty_cache()
        print(f"  🧹 GPU memory cleared")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(test_records: list,
                 predictions: dict,
                 models_run: list):
    """Print metrics report untuk semua model."""
    print(f"\n{'='*70}")
    print("TIER 2 BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  Test split: {len(test_records)} records")
    print(f"  Protocol  : Zero-shot inference, no fine-tuning")
    print(f"  Normalization: Unicode NFC + ZWNJ/ZWJ/kashida removal")

    # Tambah pred ke test_records untuk compute_metrics
    records_with_pred = []
    for r in test_records:
        r_copy = dict(r)
        rid    = r['record_id']
        for model_key in models_run:
            r_copy[model_key] = predictions.get(rid, {}).get(model_key, "")
        records_with_pred.append(r_copy)

    all_metrics = {}
    for model_key in models_run:
        m = compute_metrics(records_with_pred, model_key)
        all_metrics[model_key] = m

    # Global table
    print(f"\n  {'Model':<30} {'Params':>7} {'RC(%)':>7} "
          f"{'CER-all':>9} {'WER-all':>9} {'CER-rec':>9}")
    print(f"  {'─'*75}")
    for model_key in models_run:
        cfg = MODEL_REGISTRY[model_key]
        m   = all_metrics[model_key]
        rc  = f"{m['coverage_pct']:.1f}"
        ca  = f"{m['cer_incl_empty']:.4f}" \
              if m['cer_incl_empty'] is not None else "N/A"
        wa  = f"{m['wer_incl_empty']:.4f}" \
              if m['wer_incl_empty'] is not None else "N/A"
        cr  = f"{m['cer_excl_empty']:.4f}" \
              if m['cer_excl_empty'] is not None else "N/A"
        print(f"  {cfg['name']:<30} {cfg['params']:>7} {rc:>7} "
              f"{ca:>9} {wa:>9} {cr:>9}")

    # Per-kitab
    print(f"\n  Per-kitab CER-all:")
    print(f"  {'Kitab':<10}", end="")
    for model_key in models_run:
        name = MODEL_REGISTRY[model_key]['name'][:12]
        print(f" {name:>13}", end="")
    print()
    print(f"  {'─'*70}")

    kitabs = sorted(set(r['kitab_id'] for r in test_records))
    for kitab in kitabs:
        kitab_recs = [r for r in records_with_pred
                      if r['kitab_id'] == kitab]
        print(f"  {kitab:<10}", end="")
        for model_key in models_run:
            m   = compute_metrics(kitab_recs, model_key)
            val = f"{m['cer_incl_empty']:.4f}" \
                  if m['cer_incl_empty'] is not None else "N/A"
            print(f" {val:>13}", end="")
        print()

    # Sample predictions
    print(f"\n  Sample predictions (3 records):")
    for rec in records_with_pred[:3]:
        gt = normalize_arabic(rec.get('ground_truth', ''))[:50]
        print(f"\n  GT : {gt}")
        for model_key in models_run:
            pred = normalize_arabic(rec.get(model_key) or "")[:50]
            name = MODEL_REGISTRY[model_key]['name'][:20]
            print(f"  {name:<20}: {pred if pred else '(empty)'}")

    return all_metrics, records_with_pred


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(predictions: dict, models_run: list):
    """Simpan checkpoint predictions setelah tiap model."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, "tier2_checkpoint.json")
    ckpt = {
        "models_completed": models_run,
        "predictions"     : predictions,
    }
    with open(ckpt_path, 'w', encoding='utf-8') as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def load_checkpoint() -> tuple:
    """Load checkpoint jika ada."""
    ckpt_path = os.path.join(RESULTS_DIR, "tier2_checkpoint.json")
    if os.path.isfile(ckpt_path):
        with open(ckpt_path, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)
        models_done = ckpt.get("models_completed", [])
        predictions = ckpt.get("predictions", {})
        print(f"  ✅ Checkpoint loaded: {models_done} already done")
        return predictions, models_done
    return {}, []


def save_results(test_records: list,
                 all_metrics: dict,
                 records_with_pred: list,
                 models_run: list):
    """Simpan metrics dan predictions final."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Metrics JSON
    out = {
        "tier"       : 2,
        "description": "ViT-based OCR zero-shot on Pegon",
        "eval_split" : "test",
        "eval_n"     : len(test_records),
        "protocol"   : "zero-shot, no fine-tuning",
        "engines"    : {},
    }
    for model_key in models_run:
        cfg      = MODEL_REGISTRY[model_key]
        m        = all_metrics.get(model_key, {})
        per_kitab = {}
        for kitab in sorted(set(r['kitab_id'] for r in records_with_pred)):
            kr = [r for r in records_with_pred if r['kitab_id'] == kitab]
            per_kitab[kitab] = compute_metrics(kr, model_key)
        out["engines"][model_key] = {
            "name"       : cfg['name'],
            "hub_id"     : cfg['hub_id'],
            "params"     : cfg['params'],
            "description": cfg['description'],
            "global"     : m,
            "per_kitab"  : per_kitab,
        }

    metrics_path = os.path.join(RESULTS_DIR, "tier2_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Metrics: {metrics_path}")

    # Predictions JSON
    preds_out = []
    for r in records_with_pred:
        entry = {
            'record_id'    : r['record_id'],
            'kitab_id'     : r['kitab_id'],
            'ground_truth' : r['ground_truth'],
            'gt_normalized': normalize_arabic(r['ground_truth']),
        }
        for model_key in models_run:
            raw = r.get(model_key) or ""
            entry[model_key]                  = raw
            entry[f"{model_key}_normalized"]  = normalize_arabic(raw)
        preds_out.append(entry)

    preds_path = os.path.join(RESULTS_DIR, "tier2_predictions_test.json")
    with open(preds_path, 'w', encoding='utf-8') as f:
        json.dump(preds_out, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Predictions: {preds_path} ({len(preds_out)} records)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Model keys to run (default: all)"
    )
    parser.add_argument(
        "--skip", nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=[],
        help="Model keys to skip"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which models to run
    models_to_run = [m for m in args.models if m not in args.skip]

    print("=" * 60)
    print("CASTLE-Pegon -- Phase 05: Tier 2 OCR Benchmark")
    print("=" * 60)
    print(f"  Models : {models_to_run}")
    print(f"  Protocol: Zero-shot, no fine-tuning")

    # Load test records
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_records = json.load(f)

    # Pastikan cropped_image_path ada — ambil dari filtered JSON
    with open(FILTERED_JSON, 'r', encoding='utf-8') as f:
        filtered = json.load(f)
    crop_lookup = {r['record_id']: r.get('cropped_image_path', '')
                   for r in filtered}
    for r in test_records:
        if not r.get('cropped_image_path'):
            r['cropped_image_path'] = crop_lookup.get(r['record_id'], '')

    print(f"\n  Test records: {len(test_records)}")
    n_crop = sum(1 for r in test_records
                 if r.get('cropped_image_path')
                 and os.path.isfile(r['cropped_image_path']))
    print(f"  Crop found  : {n_crop}/{len(test_records)}")

    # Load checkpoint
    predictions, models_done = load_checkpoint()

    models_run = []

    # Run each model
    for model_key in models_to_run:
        cfg = MODEL_REGISTRY[model_key]
        print(f"\n{'='*60}")
        print(f"[{models_to_run.index(model_key)+1}/{len(models_to_run)}] "
              f"{cfg['name']}")
        print(f"{'='*60}")

        results = run_model(
            model_key, cfg, test_records, predictions
        )

        # Update predictions dict
        for record_id, pred in results.items():
            if record_id not in predictions:
                predictions[record_id] = {}
            predictions[record_id][model_key] = pred

        models_run.append(model_key)

        # Save checkpoint setelah tiap model
        save_checkpoint(predictions, models_run)
        print(f"  💾 Checkpoint saved ({model_key} done)")

    if not models_run:
        print("\n⚠️  Tidak ada model yang dijalankan.")
        return

    # Report + Save
    all_metrics, records_with_pred = print_report(
        test_records, predictions, models_run
    )
    save_results(
        test_records, all_metrics,
        records_with_pred, models_run
    )

    print(f"\n{'='*60}")
    print("PHASE 05 TIER 2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"\n  NEXT: Phase 05 Tier 3 -- General VLM")
    print(f"        Run: python src/phase05_ocr_tier3.py")
    print("=" * 60)


if __name__ == "__main__":
    main()


    # Upload dari M1
