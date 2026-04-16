"""
=============================================================================
CASTLE-Pegon -- Phase 05: Tier 4 OCR Benchmarking
=============================================================================
Engine  : Arabic-specialized OCR models
Tujuan  : Tunjukkan bahwa model yang dikhususkan untuk Arabic script
          masih gap pada Pegon karena Pegon != Arabic standar

Models:
    1. QARI-OCR v0.2   NAMAA/Qari-OCR-Arabic-0.2.2.1-VL-2B-Instruct
       Fine-tuned Qwen2-VL-2B pada Arabic printed OCR
       SOTA open-source Arabic OCR: WER=0.160, CER=0.061

    2. Arabic-HTR      sherif1313/arabic-handwritten-ocr-v2
       Fine-tuned Qwen2.5-VL-3B 4bit pada 65K Arabic handwritten samples
       CER=4.51% pada Arabic HTR standard benchmark

Protocol:
    - Zero-shot inference (no fine-tuning)
    - Prompt disesuaikan per model
    - Evaluasi pada test split (506 records)
    - Metrics: CER, WER, Coverage (RC)
    - Normalisasi Unicode sebelum metric computation

Input : data/splits/test.json
        data/processed/pegon_filtered.json
Output: results/ocr/tier4_metrics.json
        results/ocr/tier4_predictions_test.json
        results/ocr/tier4_checkpoint.json

Run:
    python src/phase05_ocr_tier4.py
    python src/phase05_ocr_tier4.py --models qari
    python src/phase05_ocr_tier4.py --skip arabic-htr
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

# Prompt per model — disesuaikan dengan training domain masing-masing
PROMPTS = {
    "qari": (
        "Read the Arabic text in this image accurately. "
        "Output only the text, nothing else."
    ),
    "arabic-htr": (
        "Please transcribe the handwritten Arabic text in this image. "
        "Output only the transcribed text, nothing else."
    ),
}

MODEL_REGISTRY = {
    "qari": {
        "name"       : "QARI-OCR-v0.2",
        "hub_id"     : "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct",
        "local_dir"  : "models/qari-ocr-v03",
        "type"       : "qwen2vl",   # base: Qwen2-VL-2B
        "params"     : "2B",
        "description": "Fine-tuned Qwen2-VL-2B on Arabic printed OCR, SOTA WER=0.160",
    },
    "arabic-htr": {
        "name"       : "Arabic-HTR-v2",
        "hub_id"     : "sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2",
        "local_dir"  : "models/arabic-htr",
        "type"       : "qwen25vl_4bit",  # base: Qwen2.5-VL-3B 4bit quantized
        "params"     : "3B-4bit",
        "description": "Fine-tuned Qwen2.5-VL-3B 4bit on 65K Arabic handwritten samples",
    },
}

REPORT_EVERY = 50

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────

import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ Device: CUDA ({torch.cuda.get_device_name(0)})")
    print(f"   VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
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
# MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_model(hub_id: str, local_dir: str):
    """Download via ModelScope (lebih reliable di AutoDL)."""
    # Cek apakah sudah ada
    if os.path.isdir(local_dir):
        files = os.listdir(local_dir)
        has_weights = any(
            f.endswith('.safetensors') or f.endswith('.bin')
            for f in files
        )
        if has_weights:
            print(f"  ✅ Already exists: {local_dir}")
            return

    print(f"  Downloading {hub_id} via ModelScope...")
    try:
        from modelscope import snapshot_download
        # ModelScope ID format mungkin berbeda — coba konversi
        ms_id = hub_id
        snapshot_download(ms_id, cache_dir=os.path.dirname(local_dir))
        print(f"  ✅ Downloaded")
    except Exception as e:
        print(f"  ⚠️  ModelScope failed: {e}")
        print(f"  Trying HuggingFace mirror...")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import snapshot_download as hf_download
        hf_download(repo_id=hub_id, local_dir=local_dir)
        print(f"  ✅ Downloaded via HF mirror")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_qwen2vl(local_dir: str):
    """Load Qwen2-VL based model (QARI-OCR)."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"  Loading Qwen2-VL from {local_dir}...")
    processor = AutoProcessor.from_pretrained(
        local_dir, trust_remote_code=True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        local_dir,
        torch_dtype = torch.bfloat16,
        device_map  = "auto",
        trust_remote_code = True,
    )
    model.eval()
    print(f"  ✅ Loaded")
    return processor, model


def load_qwen25vl_4bit(local_dir: str):
    """Load Qwen2.5-VL 4bit quantized (Arabic-HTR)."""
    from transformers import AutoProcessor

    print(f"  Loading Qwen2.5-VL 4bit from {local_dir}...")

    processor = AutoProcessor.from_pretrained(
        local_dir, trust_remote_code=True
    )

    # Coba load dengan BitsAndBytes 4bit
    try:
        from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_dir,
            quantization_config = bnb_config,
            device_map          = "auto",
            trust_remote_code   = True,
        )
        print(f"  ✅ Loaded (4bit BnB quantization)")
    except Exception as e:
        print(f"  ⚠️  4bit load failed ({e}), trying bf16...")
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_dir,
            torch_dtype = torch.bfloat16,
            device_map  = "auto",
            trust_remote_code = True,
        )
        print(f"  ✅ Loaded (bf16 fallback)")

    model.eval()
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def infer_qwen2vl(processor, model, image_path: str,
                   prompt: str) -> str:
    """Qwen2-VL inference (QARI-OCR)."""
    try:
        from qwen_vl_utils import process_vision_info
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text           = [text],
            images         = image_inputs,
            videos         = video_inputs,
            padding        = True,
            return_tensors = "pt",
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens = 256,
            )
        generated = generated[:, inputs.input_ids.shape[1]:]
        result = processor.batch_decode(
            generated,
            skip_special_tokens          = True,
            clean_up_tokenization_spaces = False,
        )[0]
        return result.strip()
    except Exception as e:
        return ""


def infer_qwen25vl(processor, model, image_path: str,
                    prompt: str) -> str:
    """Qwen2.5-VL inference (Arabic-HTR)."""
    try:
        from qwen_vl_utils import process_vision_info
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text           = [text],
            images         = image_inputs,
            videos         = video_inputs,
            padding        = True,
            return_tensors = "pt",
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens = 256,
            )
        generated = generated[:, inputs.input_ids.shape[1]:]
        result = processor.batch_decode(
            generated,
            skip_special_tokens          = True,
            clean_up_tokenization_spaces = False,
        )[0]
        return result.strip()
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
    cer_all, wer_all           = [], []
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
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_key: str,
              model_cfg: dict,
              test_records: list,
              predictions: dict) -> dict:
    print(f"\n{'─'*60}")
    print(f"Model : {model_cfg['name']} ({model_cfg['params']})")
    print(f"Hub   : {model_cfg['hub_id']}")
    print(f"Prompt: {PROMPTS[model_key][:60]}...")
    print(f"{'─'*60}")

    done = sum(
        1 for r in test_records
        if predictions.get(r['record_id'], {}).get(model_key) is not None
    )
    todo = len(test_records) - done
    print(f"  To process  : {todo}")
    print(f"  Already done: {done} (skip)")

    if todo == 0:
        print(f"  ✅ Already processed")
        return {
            r['record_id']: predictions.get(
                r['record_id'], {}
            ).get(model_key, "")
            for r in test_records
        }

    # Download
    download_model(model_cfg['hub_id'], model_cfg['local_dir'])

    # Install qwen_vl_utils jika belum ada
    try:
        import qwen_vl_utils
    except ImportError:
        os.system("pip install qwen-vl-utils -q")

    # Load model
    try:
        model_type = model_cfg['type']
        prompt     = PROMPTS[model_key]

        if model_type == "qwen2vl":
            processor, model = load_qwen2vl(model_cfg['local_dir'])
            infer_fn = lambda path: infer_qwen2vl(
                processor, model, path, prompt
            )
        elif model_type == "qwen25vl_4bit":
            processor, model = load_qwen25vl_4bit(model_cfg['local_dir'])
            infer_fn = lambda path: infer_qwen25vl(
                processor, model, path, prompt
            )
        else:
            print(f"  ❌ Unknown model type: {model_type}")
            return {}

    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        traceback.print_exc()
        return {}

    # Inference
    results = {}
    t_start = time.time()
    success = failed = 0

    for i, rec in enumerate(test_records):
        record_id = rec['record_id']

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
        except Exception:
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

    if DEVICE == "cuda":
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print(f"  🧹 GPU memory cleared")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(predictions, models_run):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "tier4_checkpoint.json"), 'w') as f:
        json.dump({
            "models_completed": models_run,
            "predictions"     : predictions,
        }, f, ensure_ascii=False, indent=2)


def load_checkpoint():
    path = os.path.join(RESULTS_DIR, "tier4_checkpoint.json")
    if os.path.isfile(path):
        with open(path) as f:
            ckpt = json.load(f)
        done = ckpt.get("models_completed", [])
        print(f"  ✅ Checkpoint loaded: {done} already done")
        return ckpt.get("predictions", {}), done
    return {}, []


# ─────────────────────────────────────────────────────────────────────────────
# REPORT & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def print_report(test_records, predictions, models_run):
    print(f"\n{'='*70}")
    print("TIER 4 BENCHMARK RESULTS — Arabic-Specialized Models")
    print(f"{'='*70}")
    print(f"  Test split: {len(test_records)} records")
    print(f"  Protocol  : Zero-shot, no fine-tuning")

    records_with_pred = []
    for r in test_records:
        r_copy = dict(r)
        rid    = r['record_id']
        for mk in models_run:
            r_copy[mk] = predictions.get(rid, {}).get(mk, "")
        records_with_pred.append(r_copy)

    all_metrics = {}
    for mk in models_run:
        all_metrics[mk] = compute_metrics(records_with_pred, mk)

    # Global table
    print(f"\n  {'Model':<25} {'Params':>8} {'RC(%)':>7} "
          f"{'CER-all':>9} {'WER-all':>9} {'CER-rec':>9}")
    print(f"  {'─'*72}")
    for mk in models_run:
        cfg = MODEL_REGISTRY[mk]
        m   = all_metrics[mk]
        rc  = f"{m['coverage_pct']:.1f}"
        ca  = f"{m['cer_incl_empty']:.4f}" if m['cer_incl_empty'] is not None else "N/A"
        wa  = f"{m['wer_incl_empty']:.4f}" if m['wer_incl_empty'] is not None else "N/A"
        cr  = f"{m['cer_excl_empty']:.4f}" if m['cer_excl_empty'] is not None else "N/A"
        print(f"  {cfg['name']:<25} {cfg['params']:>8} {rc:>7} "
              f"{ca:>9} {wa:>9} {cr:>9}")

    # Bandingkan dengan best Tier 3
    print(f"\n  Reference (best Tier 3):")
    print(f"  {'Qwen2.5-VL-7B':<25} {'7B':>8} {'100.0':>7} "
          f"{'0.5240':>9} {'1.0585':>9} {'0.5240':>9}")

    # Per-kitab
    print(f"\n  Per-kitab CER-all:")
    print(f"  {'Kitab':<10}", end="")
    for mk in models_run:
        print(f" {MODEL_REGISTRY[mk]['name'][:14]:>15}", end="")
    print()
    print(f"  {'─'*65}")
    for kitab in sorted(set(r['kitab_id'] for r in test_records)):
        kr = [r for r in records_with_pred if r['kitab_id'] == kitab]
        print(f"  {kitab:<10}", end="")
        for mk in models_run:
            m   = compute_metrics(kr, mk)
            val = f"{m['cer_incl_empty']:.4f}" \
                  if m['cer_incl_empty'] is not None else "N/A"
            print(f" {val:>15}", end="")
        print()

    # Samples
    print(f"\n  Sample predictions (3 records):")
    for rec in records_with_pred[:3]:
        gt = normalize_arabic(rec.get('ground_truth', ''))[:50]
        print(f"\n  GT : {gt}")
        for mk in models_run:
            pred = normalize_arabic(rec.get(mk) or "")[:50]
            name = MODEL_REGISTRY[mk]['name'][:22]
            print(f"  {name:<22}: {pred if pred else '(empty)'}")

    return all_metrics, records_with_pred


def save_results(test_records, all_metrics, records_with_pred, models_run):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    out = {
        "tier"       : 4,
        "description": "Arabic-specialized OCR zero-shot on Pegon",
        "eval_split" : "test",
        "eval_n"     : len(test_records),
        "engines"    : {},
    }
    for mk in models_run:
        cfg = MODEL_REGISTRY[mk]
        per_kitab = {}
        for kitab in sorted(set(r['kitab_id'] for r in records_with_pred)):
            kr = [r for r in records_with_pred if r['kitab_id'] == kitab]
            per_kitab[kitab] = compute_metrics(kr, mk)
        out["engines"][mk] = {
            "name"       : cfg['name'],
            "hub_id"     : cfg['hub_id'],
            "params"     : cfg['params'],
            "prompt"     : PROMPTS[mk],
            "description": cfg['description'],
            "global"     : all_metrics.get(mk, {}),
            "per_kitab"  : per_kitab,
        }

    metrics_path = os.path.join(RESULTS_DIR, "tier4_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Metrics: {metrics_path}")

    preds_out = []
    for r in records_with_pred:
        entry = {
            'record_id'    : r['record_id'],
            'kitab_id'     : r['kitab_id'],
            'ground_truth' : r['ground_truth'],
            'gt_normalized': normalize_arabic(r['ground_truth']),
        }
        for mk in models_run:
            raw = r.get(mk) or ""
            entry[mk]                 = raw
            entry[f"{mk}_normalized"] = normalize_arabic(raw)
        preds_out.append(entry)

    preds_path = os.path.join(RESULTS_DIR, "tier4_predictions_test.json")
    with open(preds_path, 'w') as f:
        json.dump(preds_out, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Predictions: {len(preds_out)} records")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=list(MODEL_REGISTRY.keys()),
                        default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--skip", nargs="+",
                        choices=list(MODEL_REGISTRY.keys()),
                        default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    models_to_run = [m for m in args.models if m not in args.skip]

    print("=" * 60)
    print("CASTLE-Pegon -- Phase 05: Tier 4 OCR Benchmark")
    print("=" * 60)
    print("  Arabic-Specialized Models — Zero-Shot on Pegon")
    print(f"  Models: {models_to_run}")

    with open(TEST_JSON) as f:
        test_records = json.load(f)

    with open(FILTERED_JSON) as f:
        filtered = json.load(f)
    crop_lookup = {r['record_id']: r.get('cropped_image_path', '')
                   for r in filtered}
    for r in test_records:
        if not r.get('cropped_image_path'):
            r['cropped_image_path'] = crop_lookup.get(r['record_id'], '')

    n_crop = sum(1 for r in test_records
                 if r.get('cropped_image_path')
                 and os.path.isfile(r['cropped_image_path']))
    print(f"\n  Test records: {len(test_records)}")
    print(f"  Crop found  : {n_crop}/{len(test_records)}")

    predictions, _ = load_checkpoint()
    models_run = []

    for model_key in models_to_run:
        cfg = MODEL_REGISTRY[model_key]
        print(f"\n{'='*60}")
        print(f"[{models_to_run.index(model_key)+1}/{len(models_to_run)}] "
              f"{cfg['name']}")
        print(f"{'='*60}")

        results = run_model(model_key, cfg, test_records, predictions)

        for rid, pred in results.items():
            if rid not in predictions:
                predictions[rid] = {}
            predictions[rid][model_key] = pred

        models_run.append(model_key)
        save_checkpoint(predictions, models_run)
        print(f"  💾 Checkpoint saved ({model_key} done)")

    if not models_run:
        print("\n⚠️  Tidak ada model yang dijalankan.")
        return

    all_metrics, records_with_pred = print_report(
        test_records, predictions, models_run
    )
    save_results(test_records, all_metrics, records_with_pred, models_run)

    print(f"\n{'='*60}")
    print("PHASE 05 TIER 4 COMPLETE")
    print(f"{'='*60}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"\n  NEXT: Phase 05 Tier 5 -- Post-OCR Correction (ByT5)")
    print(f"        Run: python src/phase05_ocr_tier5.py")
    print("=" * 60)


if __name__ == "__main__":
    main()