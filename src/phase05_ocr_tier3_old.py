"""
=============================================================================
CASTLE-Pegon -- Phase 05: Tier 3 OCR Benchmarking
=============================================================================
Engine  : General-purpose Vision-Language Models (VLM)
Tujuan  : Tunjukkan bahwa VLM besar perform lebih baik karena broader
          pre-training, tapi masih belum cukup untuk Pegon

Models:
    1. Qwen2.5-VL-3B    Qwen/Qwen2.5-VL-3B-Instruct
    2. Qwen2.5-VL-7B    Qwen/Qwen2.5-VL-7B-Instruct
    3. InternVL2.5-4B   OpenGVLab/InternVL2_5-4B

Protocol:
    - Zero-shot inference (no fine-tuning)
    - Prompt: "Please transcribe the Arabic/Pegon handwritten text in this
               image. Output only the transcribed text, nothing else."
    - Evaluasi pada test split (506 records)
    - Metrics: CER, WER, Coverage (RC)
    - Normalisasi Unicode sebelum metric computation

Input : data/splits/test.json
        data/processed/pegon_filtered.json (untuk cropped_image_path)
Output: results/ocr/tier3_metrics.json
        results/ocr/tier3_predictions_test.json
        results/ocr/tier3_checkpoint.json

Run:
    python src/phase05_ocr_tier3.py
    python src/phase05_ocr_tier3.py --models qwen-3b qwen-7b
    python src/phase05_ocr_tier3.py --skip internvl-4b
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
import os
from huggingface_hub import snapshot_download

# Add at the top of your file or in the model loading function
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Faster downloads


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

# OCR prompt — sama untuk semua VLM
OCR_PROMPT = (
    "Please transcribe the Arabic/Pegon handwritten text in this image. "
    "Output only the transcribed text, nothing else."
)

MODEL_REGISTRY = {
    "qwen-3b": {
        "name"       : "Qwen2.5-VL-3B",
        "hub_id"     : "Qwen/Qwen2.5-VL-3B-Instruct",
        "local_dir"  : "models/qwen-3b/Qwen/Qwen2.5-VL-3B-Instruct",
        "type"       : "qwen",
        "params"     : "3B",
        "description": "Qwen2.5 VL 3B Instruct, general multimodal",
    },
    "qwen-7b": {
        "name"       : "Qwen2.5-VL-7B",
        "hub_id"     : "Qwen/Qwen2.5-VL-7B-Instruct",
        "local_dir"  : "models/qwen-7b/Qwen/Qwen2.5-VL-7B-Instruct",
        "type"       : "qwen",
        "params"     : "7B",
        "description": "Qwen2.5 VL 7B Instruct, general multimodal",
    },
    "internvl-4b": {
        "name"       : "InternVL2.5-4B",
        "hub_id"     : "OpenGVLab/InternVL2_5-4B",
        "local_dir"  : "models/internvl-4b/OpenGVLab/InternVL2_5-4B",
        "type"       : "internvl",
        "params"     : "4B",
        "description": "InternVL2.5 4B, strong OCRBench performance",
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
    """Download model dari HuggingFace mirror jika belum ada."""
    # Cek apakah sudah ada weights
    has_weights = any(
        os.path.isfile(os.path.join(local_dir, f))
        for f in os.listdir(local_dir)
        if f.endswith('.safetensors') or f.endswith('.bin')
    ) if os.path.isdir(local_dir) else False

    if has_weights:
        print(f"  ✅ Model already exists: {local_dir}")
        return

    print(f"  Downloading {hub_id} → {local_dir}...")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=hub_id, local_dir=local_dir)
    print(f"  ✅ Downloaded: {local_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_qwen(local_dir: str):
    """Load Qwen2.5-VL model."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print(f"  Loading Qwen2.5-VL from {local_dir}...")
    processor = AutoProcessor.from_pretrained(
        local_dir, trust_remote_code=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_dir,
        torch_dtype = torch.bfloat16,
        device_map  = "auto",
        trust_remote_code = True,
    )
    model.eval()
    print(f"  ✅ Loaded")
    return processor, model


# def load_internvl(local_dir: str):
#     """Load InternVL2.5 model."""
#     from transformers import AutoTokenizer, AutoModel
#     import torchvision.transforms as T
#     from torchvision.transforms.functional import InterpolationMode

#     print(f"  Loading InternVL2.5 from {local_dir}...")
#     tokenizer = AutoTokenizer.from_pretrained(
#         local_dir, trust_remote_code=True, use_fast=False
#     )
#     model = AutoModel.from_pretrained(
#         local_dir,
#         torch_dtype       = torch.bfloat16,
#         device_map        = "auto",
#         trust_remote_code = True,
#     )
#     model.eval()
#     print(f"  ✅ Loaded")
#     return tokenizer, model

def load_internvl(local_dir: str):
    """Load InternVL2.5 model optimized for RTX 5090."""
    from transformers import AutoTokenizer, AutoModel
    import torch

    print(f"  Loading InternVL2.5 from {local_dir}...")
    
    # ---------------------------------------------------------
    # 🛠️ PATCH BUG INTERNVL2.5 (Wajib ada)
    # ---------------------------------------------------------
    _orig_linspace = torch.linspace
    def patched_linspace(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return _orig_linspace(*args, **kwargs)
    torch.linspace = patched_linspace
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        local_dir, trust_remote_code=True
    )
    
    # Langsung tembak ke GPU 0
    model = AutoModel.from_pretrained(
        local_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0", 
        trust_remote_code=True,
    ).eval()
    
    print(f"  ✅ Loaded to {torch.cuda.get_device_name(0)}")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def infer_qwen(processor, model, image_path: str) -> str:
    """Qwen2.5-VL inference."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": OCR_PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text          = [text],
            images        = image_inputs,
            videos        = video_inputs,
            padding       = True,
            return_tensors= "pt",
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens = 256,
            )
        # Trim input tokens
        generated = generated[:, inputs.input_ids.shape[1]:]
        result = processor.batch_decode(
            generated,
            skip_special_tokens     = True,
            clean_up_tokenization_spaces = False,
        )[0]
        return result.strip()
    except Exception as e:
        return ""


def infer_internvl(tokenizer, model, image_path: str) -> str:
    """InternVL2.5 inference."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from PIL import Image

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def build_transform(input_size=448):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def load_image(path, input_size=448):
        img = Image.open(path).convert('RGB')
        transform = build_transform(input_size)
        return transform(img).unsqueeze(0)

    try:
        pixel_values = load_image(image_path).to(
            dtype=torch.bfloat16, device=DEVICE
        )
        question     = f"<image>\n{OCR_PROMPT}"
        generation_config = dict(max_new_tokens=256, do_sample=False)

        with torch.no_grad():
            response = model.chat(
                tokenizer, pixel_values, question, 
                generation_config=dict(
                    max_new_tokens=256, 
                    do_sample=False,
                    # 👇 Tambahkan baris ini di config generate
                    pad_token_id=tokenizer.eos_token_id
                )
            )
        return str(response).strip()
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
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_key: str,
              model_cfg: dict,
              test_records: list,
              predictions: dict) -> dict:
    print(f"\n{'─'*60}")
    print(f"Model: {model_cfg['name']} ({model_cfg['params']})")
    print(f"Hub  : {model_cfg['hub_id']}")
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
            r['record_id']: predictions.get(r['record_id'], {}).get(model_key, "")
            for r in test_records
        }

    # Download jika belum ada
    download_model(model_cfg['hub_id'], model_cfg['local_dir'])

    # Load model
    try:
        model_type = model_cfg['type']
        if model_type == "qwen":
            # Install qwen_vl_utils jika belum ada
            try:
                import qwen_vl_utils
            except ImportError:
                os.system("pip install qwen-vl-utils -q")
            processor, model = load_qwen(model_cfg['local_dir'])
            infer_fn = lambda path: infer_qwen(processor, model, path)
        elif model_type == "internvl":
            tokenizer, model = load_internvl(model_cfg['local_dir'])
            infer_fn = lambda path: infer_internvl(tokenizer, model, path)
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

    # GPU cleanup
    if DEVICE == "cuda":
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print(f"  🧹 GPU memory cleared")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORT & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def print_report(test_records, predictions, models_run):
    print(f"\n{'='*70}")
    print("TIER 3 BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  Test split: {len(test_records)} records")
    print(f"  Protocol  : Zero-shot VLM inference")
    print(f"  Prompt    : \"{OCR_PROMPT[:60]}...\"")

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
    print(f"\n  {'Model':<25} {'Params':>6} {'RC(%)':>7} "
          f"{'CER-all':>9} {'WER-all':>9} {'CER-rec':>9}")
    print(f"  {'─'*70}")
    for mk in models_run:
        cfg = MODEL_REGISTRY[mk]
        m   = all_metrics[mk]
        rc  = f"{m['coverage_pct']:.1f}"
        ca  = f"{m['cer_incl_empty']:.4f}" if m['cer_incl_empty'] is not None else "N/A"
        wa  = f"{m['wer_incl_empty']:.4f}" if m['wer_incl_empty'] is not None else "N/A"
        cr  = f"{m['cer_excl_empty']:.4f}" if m['cer_excl_empty'] is not None else "N/A"
        print(f"  {cfg['name']:<25} {cfg['params']:>6} {rc:>7} "
              f"{ca:>9} {wa:>9} {cr:>9}")

    # Per-kitab
    print(f"\n  Per-kitab CER-all:")
    print(f"  {'Kitab':<10}", end="")
    for mk in models_run:
        print(f" {MODEL_REGISTRY[mk]['name'][:12]:>13}", end="")
    print()
    print(f"  {'─'*60}")
    for kitab in sorted(set(r['kitab_id'] for r in test_records)):
        kr = [r for r in records_with_pred if r['kitab_id'] == kitab]
        print(f"  {kitab:<10}", end="")
        for mk in models_run:
            m   = compute_metrics(kr, mk)
            val = f"{m['cer_incl_empty']:.4f}" \
                  if m['cer_incl_empty'] is not None else "N/A"
            print(f" {val:>13}", end="")
        print()

    # Samples
    print(f"\n  Sample predictions (3 records):")
    for rec in records_with_pred[:3]:
        gt = normalize_arabic(rec.get('ground_truth', ''))[:50]
        print(f"\n  GT : {gt}")
        for mk in models_run:
            pred = normalize_arabic(rec.get(mk) or "")[:50]
            name = MODEL_REGISTRY[mk]['name'][:20]
            print(f"  {name:<20}: {pred if pred else '(empty)'}")

    return all_metrics, records_with_pred


def save_checkpoint(predictions, models_run):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "tier3_checkpoint.json"), 'w') as f:
        json.dump({
            "models_completed": models_run,
            "predictions"     : predictions,
        }, f, ensure_ascii=False, indent=2)


def load_checkpoint():
    path = os.path.join(RESULTS_DIR, "tier3_checkpoint.json")
    if os.path.isfile(path):
        with open(path) as f:
            ckpt = json.load(f)
        done = ckpt.get("models_completed", [])
        print(f"  ✅ Checkpoint loaded: {done} already done")
        return ckpt.get("predictions", {}), done
    return {}, []


def save_results(test_records, all_metrics, records_with_pred, models_run):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    out = {
        "tier"       : 3,
        "description": "General-purpose VLM zero-shot on Pegon",
        "eval_split" : "test",
        "eval_n"     : len(test_records),
        "prompt"     : OCR_PROMPT,
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
            "description": cfg['description'],
            "global"     : all_metrics.get(mk, {}),
            "per_kitab"  : per_kitab,
        }

    with open(os.path.join(RESULTS_DIR, "tier3_metrics.json"), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Metrics: {RESULTS_DIR}/tier3_metrics.json")

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

    with open(os.path.join(RESULTS_DIR, "tier3_predictions_test.json"), 'w') as f:
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
    print("CASTLE-Pegon -- Phase 05: Tier 3 OCR Benchmark (VLM)")
    print("=" * 60)
    print(f"  Models  : {models_to_run}")
    print(f"  Protocol: Zero-shot, no fine-tuning")

    # Load test records
    with open(TEST_JSON) as f:
        test_records = json.load(f)

    # Fix cropped_image_path dari filtered JSON
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

    predictions, models_done = load_checkpoint()
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
    print("PHASE 05 TIER 3 COMPLETE")
    print(f"{'='*60}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"\n  NEXT: Phase 05 Tier 4 -- Arabic-specialized models")
    print(f"        Run: python src/phase05_ocr_tier4.py")
    print("=" * 60)


if __name__ == "__main__":
    main()