"""
=============================================================================
CASTLE-Pegon -- Phase 05: GPT-4o OCR untuk Train Split
=============================================================================
Tujuan  : Jalankan GPT-4o pada 1936 train records
Input   : data/splits/train.json
Output  : data/splits/train.json (updated dengan field ocr_gpt4o)
          results/ocr/gpt4o_train_checkpoint.json
          results/ocr/gpt4o_train_metrics.json

Estimasi: ~1936 records × $0.012 = ~$23 USD
          ~1936 records × 3 detik = ~96 menit

Run: python src/phase05_ocr_gpt4o_train.py

Notes:
  - Resume-friendly: skip records yang sudah ada di checkpoint
  - Checkpoint disimpan setiap 20 records
  - Jalankan di M1 dengan VPN jika perlu
=============================================================================
"""

import os
import sys
import json
import time
import base64
import unicodedata
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPLITS_DIR, RESULTS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_JSON  = os.path.join(SPLITS_DIR, "train.json")
RESULTS_OCR = os.path.join(RESULTS_DIR, "ocr")
CHECKPOINT  = os.path.join(RESULTS_OCR, "gpt4o_train_checkpoint.json")

MODEL       = "gpt-4o"
MAX_TOKENS  = 256
RETRY_MAX   = 3
RETRY_DELAY = 5
SAVE_EVERY  = 20

OCR_PROMPT = (
    "Please transcribe the Arabic/Pegon handwritten text in this image. "
    "Output only the transcribed text, nothing else."
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\u0640", "")
    return " ".join(text.split()).strip()


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


def compute_cer(pred, gt):
    pred = normalize_arabic(pred)
    gt   = normalize_arabic(gt)
    if not gt:
        return 0.0
    return edit_distance(pred, gt) / len(gt)


def encode_image(image_path: str) -> tuple[str, str]:
    """Returns (base64_string, mime_type)."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
    return data, mime


def run_gpt4o(client: OpenAI, image_path: str) -> str:
    for attempt in range(RETRY_MAX):
        try:
            b64, mime = encode_image(image_path)
            response  = client.chat.completions.create(
                model      = MODEL,
                max_tokens = MAX_TOKENS,
                messages   = [{
                    "role": "user",
                    "content": [
                        {
                            "type"     : "image_url",
                            "image_url": {
                                "url"   : f"data:{mime};base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    ⚠️  Attempt {attempt+1}/{RETRY_MAX}: {e}")
            if attempt < RETRY_MAX - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if os.path.isfile(CHECKPOINT):
        with open(CHECKPOINT) as f:
            ckpt = json.load(f)
        done = ckpt.get("predictions", {})
        print(f"  ✅ Checkpoint: {len(done)} already done")
        return done
    return {}


def save_checkpoint(predictions: dict):
    os.makedirs(RESULTS_OCR, exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump({"model": MODEL, "predictions": predictions},
                  f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_OCR, exist_ok=True)

    print("=" * 55)
    print("CASTLE-Pegon -- GPT-4o OCR for Train Split")
    print("=" * 55)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY tidak ditemukan.")
        print("   export OPENAI_API_KEY=sk-...")
        return

    client = OpenAI(api_key=api_key)
    print(f"  Model : {MODEL}")

    # Load train records
    with open(TRAIN_JSON) as f:
        train_records = json.load(f)

    # Cek crop paths
    n_crop = sum(
        1 for r in train_records
        if r.get("cropped_image_path")
        and os.path.isfile(r["cropped_image_path"])
    )
    print(f"\n  Train records : {len(train_records)}")
    print(f"  Crop found    : {n_crop}/{len(train_records)}")

    if n_crop == 0:
        print("❌ Tidak ada crop image ditemukan.")
        return

    # Load checkpoint
    predictions = load_checkpoint()

    # Records yang perlu diproses
    to_process = [
        r for r in train_records
        if r["record_id"] not in predictions
        and r.get("cropped_image_path")
        and os.path.isfile(r["cropped_image_path"])
    ]

    print(f"\n  To process  : {len(to_process)}")
    print(f"  Already done: {len(predictions)}")
    est_cost = len(to_process) * 0.012
    est_min  = len(to_process) * 3 / 60
    print(f"\n  Estimated cost: ~${est_cost:.2f} USD")
    print(f"  Estimated time: ~{est_min:.0f} minutes")

    confirm = input("\n  Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    # Run GPT-4o
    print(f"\nRunning GPT-4o on {len(to_process)} records...")
    t_start = time.time()
    success = failed = 0

    for i, rec in enumerate(to_process):
        record_id = rec["record_id"]
        img_path  = rec["cropped_image_path"]

        text = run_gpt4o(client, img_path)

        predictions[record_id] = text
        if text:
            success += 1
        else:
            failed += 1

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(to_process):
            elapsed   = time.time() - t_start
            per_img   = elapsed / (i + 1)
            remaining = (len(to_process) - i - 1) * per_img
            print(f"  [{i+1}/{len(to_process)}] "
                  f"ok={success} fail={failed} | "
                  f"{elapsed:.0f}s | ETA {remaining:.0f}s")
            save_checkpoint(predictions)

    elapsed = time.time() - t_start
    print(f"\n  ✅ Done: {success} ok, {failed} failed | {elapsed:.1f}s")

    # Update train.json
    print(f"\nUpdating train.json...")
    matched = 0
    for r in train_records:
        rid = r["record_id"]
        if rid in predictions:
            r["ocr_gpt4o"] = predictions[rid]
            matched += 1
        else:
            r["ocr_gpt4o"] = ""

    with open(TRAIN_JSON, "w") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    print(f"  ✅ train.json updated ({matched} records with ocr_gpt4o)")

    # Compute metrics
    cer_scores = []
    n_empty    = 0
    for r in train_records:
        pred = r.get("ocr_gpt4o", "")
        gt   = r.get("ground_truth", "")
        if not gt:
            continue
        if not pred:
            n_empty += 1
        cer_scores.append(compute_cer(pred, gt))

    mean_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    coverage = (len(cer_scores) - n_empty) / len(cer_scores) * 100

    metrics = {
        "model"       : MODEL,
        "split"       : "train",
        "n"           : len(cer_scores),
        "n_empty"     : n_empty,
        "coverage_pct": round(coverage, 1),
        "cer"         : round(mean_cer, 4),
    }
    with open(os.path.join(RESULTS_OCR, "gpt4o_train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*55}")
    print("GPT-4o TRAIN COMPLETE")
    print(f"{'='*55}")
    print(f"  CER      : {mean_cer:.4f}")
    print(f"  Coverage : {coverage:.1f}%")
    print(f"  Empty    : {n_empty}")
    print(f"\n  NEXT:")
    print(f"  1. Upload train.json ke AutoDL")
    print(f"  2. python src/phase07_augment.py  (regenerate dengan input=ocr_gpt4o)")
    print(f"  3. python src/phase08_byt5.py --mode glyph")
    print("=" * 55)


if __name__ == "__main__":
    main()