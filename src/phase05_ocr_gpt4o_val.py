"""
=============================================================================
CASTLE-Pegon -- Phase 05: GPT-4o OCR untuk Val Set
=============================================================================
Tujuan  : Jalankan GPT-4o pada 359 val records untuk mengisi ocr_gpt4o
Input   : data/splits/val.json
Output  : data/splits/val.json (updated dengan ocr_gpt4o)
          results/ocr/gpt4o_val_checkpoint.json
          results/ocr/gpt4o_val_metrics.json

Run: python src/phase05_ocr_gpt4o_val.py

Notes:
  - Resume-friendly: skip records yang sudah ada ocr_gpt4o
  - Checkpoint disimpan setiap 10 records
  - Estimasi biaya: ~$3-5 USD untuk 359 images
  - Gunakan VPN jika di server dengan firewall
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
from config import SPLITS_DIR, RESULTS_DIR, LOGS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

VAL_JSON      = os.path.join(SPLITS_DIR, "val.json")
RESULTS_OCR   = os.path.join(RESULTS_DIR, "ocr")
CHECKPOINT    = os.path.join(RESULTS_OCR, "gpt4o_val_checkpoint.json")

MODEL         = "gpt-4o"
MAX_TOKENS    = 256
RETRY_MAX     = 3
RETRY_DELAY   = 5    # seconds
SAVE_EVERY    = 10   # checkpoint setiap N records

OCR_PROMPT = (
    "Please transcribe the Arabic/Pegon handwritten text in this image. "
    "Output only the transcribed text, nothing else."
)

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


def compute_cer(pred, gt):
    pred = normalize_arabic(pred)
    gt   = normalize_arabic(gt)
    if not gt:
        return 0.0
    return edit_distance(pred, gt) / len(gt)


def compute_wer(pred, gt):
    pred = normalize_arabic(pred)
    gt   = normalize_arabic(gt)
    pw, gw = pred.split(), gt.split()
    if not gw:
        return 0.0
    return edit_distance(pw, gw) / len(gw)


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def encode_image(image_path: str) -> str:
    """Encode image ke base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_gpt4o(client: OpenAI, image_path: str,
              retry_max: int = RETRY_MAX) -> str:
    """
    Jalankan GPT-4o OCR pada satu image.
    Returns: teks hasil OCR atau "" jika gagal.
    """
    for attempt in range(retry_max):
        try:
            b64 = encode_image(image_path)
            ext = Path(image_path).suffix.lower().lstrip(".")
            if ext == "jpg":
                ext = "jpeg"
            mime = f"image/{ext}"

            response = client.chat.completions.create(
                model      = MODEL,
                max_tokens = MAX_TOKENS,
                messages   = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url"   : f"data:{mime};base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": OCR_PROMPT,
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"    ⚠️  Attempt {attempt+1}/{retry_max} failed: {e}")
            if attempt < retry_max - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint():
    if os.path.isfile(CHECKPOINT):
        with open(CHECKPOINT) as f:
            ckpt = json.load(f)
        done = ckpt.get("predictions", {})
        print(f"  ✅ Checkpoint loaded: {len(done)} already done")
        return done
    return {}


def save_checkpoint(predictions: dict):
    os.makedirs(RESULTS_OCR, exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump({"predictions": predictions}, f,
                  ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_OCR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 55)
    print("CASTLE-Pegon -- GPT-4o OCR for Val Set")
    print("=" * 55)

    # Init OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY tidak ditemukan.")
        print("   Set dengan: export OPENAI_API_KEY=sk-...")
        return

    client = OpenAI(api_key=api_key)
    print(f"  Model  : {MODEL}")
    print(f"  Prompt : {OCR_PROMPT[:50]}...")

    # Load val records
    with open(VAL_JSON) as f:
        val_records = json.load(f)
    print(f"\n  Val records: {len(val_records)}")

    # Cek crop paths
    n_crop = sum(
        1 for r in val_records
        if r.get("cropped_image_path")
        and os.path.isfile(r["cropped_image_path"])
    )
    print(f"  Crop found : {n_crop}/{len(val_records)}")

    if n_crop == 0:
        print("❌ Tidak ada crop image ditemukan.")
        print("   Pastikan cropped_image_path benar di val.json")
        return

    # Load checkpoint
    predictions = load_checkpoint()

    # Hitung yang perlu diproses
    to_process = [
        r for r in val_records
        if r["record_id"] not in predictions
        and r.get("cropped_image_path")
        and os.path.isfile(r["cropped_image_path"])
    ]
    print(f"\n  To process  : {len(to_process)}")
    print(f"  Already done: {len(predictions)}")
    print(f"\n  Estimated cost: ~${len(to_process) * 0.012:.2f} USD")
    print(f"  Estimated time: ~{len(to_process) * 3 // 60} minutes")

    # Confirm
    confirm = input("\n  Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    # Run GPT-4o
    print(f"\nRunning GPT-4o...")
    t_start = time.time()
    success = failed = 0

    for i, rec in enumerate(to_process):
        record_id = rec["record_id"]
        img_path  = rec["cropped_image_path"]

        text = run_gpt4o(client, img_path)

        if text:
            predictions[record_id] = text
            success += 1
        else:
            predictions[record_id] = ""
            failed += 1

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
            elapsed  = time.time() - t_start
            per_img  = elapsed / (i + 1)
            remaining= (len(to_process) - i - 1) * per_img
            print(f"  [{i+1}/{len(to_process)}] "
                  f"ok={success} fail={failed} | "
                  f"{elapsed:.0f}s | ETA {remaining:.0f}s")
            # Save checkpoint
            save_checkpoint(predictions)

    elapsed = time.time() - t_start
    print(f"\n  ✅ Done: {success} ok, {failed} failed | {elapsed:.1f}s")

    # Update val.json dengan ocr_gpt4o
    print(f"\nUpdating val.json...")
    for r in val_records:
        rid = r["record_id"]
        r["ocr_gpt4o"] = predictions.get(rid, "")

    with open(VAL_JSON, "w") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)
    print(f"  ✅ val.json updated")

    # Compute metrics
    print(f"\nComputing metrics...")
    cer_scores, wer_scores = [], []
    n_empty = 0

    for r in val_records:
        pred = r.get("ocr_gpt4o", "")
        gt   = r.get("ground_truth", "")
        if not gt:
            continue
        if not pred:
            n_empty += 1
        cer_scores.append(compute_cer(pred, gt))
        wer_scores.append(compute_wer(pred, gt))

    mean_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    mean_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 1.0
    coverage = (len(cer_scores) - n_empty) / len(cer_scores) * 100

    metrics = {
        "model"         : MODEL,
        "split"         : "val",
        "n"             : len(cer_scores),
        "n_empty"       : n_empty,
        "coverage_pct"  : round(coverage, 1),
        "cer"           : round(mean_cer, 4),
        "wer"           : round(mean_wer, 4),
    }

    metrics_path = os.path.join(RESULTS_OCR, "gpt4o_val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*55}")
    print("GPT-4o VAL SET COMPLETE")
    print(f"{'='*55}")
    print(f"  CER      : {mean_cer:.4f}")
    print(f"  WER      : {mean_wer:.4f}")
    print(f"  Coverage : {coverage:.1f}%")
    print(f"  Empty    : {n_empty}")
    print(f"\n  NEXT: python src/phase07_augment.py")
    print("=" * 55)


if __name__ == "__main__":
    main()