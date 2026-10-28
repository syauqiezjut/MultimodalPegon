"""
=============================================================================
CASTLE-Pegon -- Phase 05: Tier 3 GPT OCR Benchmarking
=============================================================================
Engine  : GPT-4o / GPT-4o-mini (via OpenAI API)
Tujuan  : Benchmark proprietary VLM on Pegon as Tier 3 comparison

Protocol:
    - Zero-shot inference (no fine-tuning)
    - Input: cropped line images (already available)
    - Evaluasi pada test split (506 records)
    - Metrics: CER, WER, Coverage
    - Resume-friendly: skip records yang sudah ada hasil

Run:
    # Estimasi biaya dulu:
    python src/phase05_ocr_tier3_gpt.py --mode estimate

    # Test 5 records:
    python src/phase05_ocr_tier3_gpt.py --mode test

    # Full run:
    python src/phase05_ocr_tier3_gpt.py --mode full

    # Pakai gpt-4o (lebih mahal, lebih akurat):
    python src/phase05_ocr_tier3_gpt.py --mode full --model gpt-4o

    # Limit jumlah:
    python src/phase05_ocr_tier3_gpt.py --mode full --limit 100
=============================================================================
"""

import os
import sys
import json
import math
import time
import base64
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime

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

# Model default — gpt-4o-mini jauh lebih murah untuk benchmarking
DEFAULT_MODEL = "gpt-4o-mini"

# API key dari environment variable (JANGAN hardcode!)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Rate limiting
REQUEST_DELAY = 4.0    # detik antar request
SAVE_EVERY_N  = 25     # auto-save checkpoint setiap N records

# Prompt — konsisten dengan Tier 3 VLM
OCR_PROMPT = (
    "Please transcribe the Arabic/Pegon handwritten text in this image. Output only the transcribed text, nothing else."
)

# Pricing (per 1K tokens, USD) — update jika berubah
PRICING = {
    "gpt-4o": {
        "input_text" : 0.005,
        "input_image": 0.01275,
        "output"     : 0.015,
    },
    "gpt-4o-mini": {
        "input_text" : 0.000150,
        "input_image": 0.001275,
        "output"     : 0.000600,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION (sama dengan tier lain)
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
# IMAGE ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def image_to_base64(image_path: str) -> str:
    """Encode cropped line image ke base64 PNG."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def estimate_image_tokens(image_path: str) -> int:
    """Estimasi token untuk satu image (OpenAI detail=high)."""
    from PIL import Image
    img = Image.open(image_path)
    w_px, h_px = img.size

    # OpenAI algorithm: fit 2048x2048 → shortest=768 → count 512x512 tiles
    max_dim = max(h_px, w_px)
    if max_dim > 2048:
        scale = 2048 / max_dim
        h_px, w_px = int(h_px * scale), int(w_px * scale)
    min_dim = min(h_px, w_px)
    if min_dim > 768:
        scale = 768 / min_dim
        h_px, w_px = int(h_px * scale), int(w_px * scale)

    tiles = math.ceil(h_px / 512) * math.ceil(w_px / 512)
    return tiles * 170 + 85


# ─────────────────────────────────────────────────────────────────────────────
# COST ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_cost_per_record(image_path: str, model: str) -> float:
    """Estimasi biaya satu record."""
    price = PRICING.get(model, PRICING["gpt-4o-mini"])
    try:
        img_tokens = estimate_image_tokens(image_path)
    except Exception:
        img_tokens = 255  # fallback estimate
    prompt_tokens = 150   # rough estimate for OCR_PROMPT
    output_tokens = 50    # average Arabic line output

    return (
        (prompt_tokens / 1000) * price["input_text"]  +
        (img_tokens    / 1000) * price["input_image"] +
        (output_tokens / 1000) * price["output"]
    )


def print_cost_estimate(pending_records: list, model: str):
    """Tampilkan estimasi biaya."""
    if not pending_records:
        print("  ✅ Semua record sudah diproses.")
        return 0, 0.0

    costs = []
    for r in pending_records[:50]:  # sample 50 for estimate
        img_path = r.get("cropped_image_path", "")
        if img_path and os.path.isfile(img_path):
            costs.append(estimate_cost_per_record(img_path, model))

    if not costs:
        print("  ⚠️  Tidak ada crop images yang valid.")
        return 0, 0.0

    avg_cost   = sum(costs) / len(costs)
    total_cost = avg_cost * len(pending_records)

    price = PRICING.get(model, PRICING["gpt-4o-mini"])
    print(f"\n  {'─'*50}")
    print(f"  💰 ESTIMASI BIAYA")
    print(f"  {'─'*50}")
    print(f"  Model           : {model}")
    print(f"  Records pending : {len(pending_records)}")
    print(f"  Avg cost/record : ${avg_cost:.6f}")
    print(f"  Total estimated : ${total_cost:.4f}")
    print(f"  {'─'*50}")

    # Comparison table
    for m in ["gpt-4o-mini", "gpt-4o"]:
        p = PRICING[m]
        c = avg_cost * (sum(p.values()) / sum(PRICING[model].values()))
        t = c * len(pending_records)
        marker = " ◀" if m == model else ""
        print(f"  {m:<15}: ${t:.4f} total  (${c:.6f}/rec){marker}")

    print(f"  {'─'*50}")
    return len(pending_records), total_cost


# ─────────────────────────────────────────────────────────────────────────────
# GPT API CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_gpt_ocr(client, image_path: str, model: str) -> tuple:
    """
    Call GPT API untuk satu cropped line image.
    Returns: (transcription_text, usage_dict)
    """
    b64 = image_to_base64(image_path)

    # Detect image format
    ext = Path(image_path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"
            }.get(ext.lstrip("."), "image/png")

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url"   : f"data:{mime};base64,{b64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": OCR_PROMPT
                }
            ]
        }],
        max_tokens=200,
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip().strip('"\'`')
    usage = {
        "prompt_tokens"    : response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens"     : response.usage.total_tokens,
    }
    return text, usage


def compute_actual_cost(usage: dict, model: str) -> float:
    price = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (
        (usage["prompt_tokens"]     / 1000) * price["input_text"]  +
        (usage["completion_tokens"] / 1000) * price["output"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS (sama dengan tier lain)
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
        wer = edit_distance(pred.split(), gt.split()) / max(len(gt.split()), 1)

        cer_all.append(cer)
        wer_all.append(wer)
        if pred and not pred.startswith("["):
            cer_nonempty.append(cer)
            wer_nonempty.append(wer)
        else:
            n_empty += 1

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "n_total"        : len(cer_all),
        "n_empty_pred"   : n_empty,
        "n_nonempty_pred": len(cer_nonempty),
        "coverage_pct"   : round(len(cer_nonempty) / max(len(cer_all), 1) * 100, 1),
        "cer_incl_empty" : avg(cer_all),
        "wer_incl_empty" : avg(wer_all),
        "cer_excl_empty" : avg(cer_nonempty),
        "wer_excl_empty" : avg(wer_nonempty),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def get_checkpoint_path(model: str) -> str:
    return os.path.join(RESULTS_DIR, f"tier3_gpt_{model.replace('-','_')}_checkpoint.json")


def save_checkpoint(predictions: dict, model: str, cost_total: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(get_checkpoint_path(model), "w") as f:
        json.dump({
            "model"      : model,
            "predictions": predictions,
            "cost_total" : cost_total,
            "timestamp"  : datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)


def load_checkpoint(model: str) -> tuple:
    path = get_checkpoint_path(model)
    if os.path.isfile(path):
        with open(path) as f:
            ckpt = json.load(f)
        preds = ckpt.get("predictions", {})
        cost  = ckpt.get("cost_total", 0.0)
        print(f"  ✅ Checkpoint loaded: {len(preds)} records, ${cost:.4f} spent")
        return preds, cost
    return {}, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_gpt_ocr(test_records: list, model: str,
                mode: str, limit: int = None):
    """Run GPT OCR on test records."""
    import openai

    if not OPENAI_API_KEY:
        print("\n  ❌ OPENAI_API_KEY not set!")
        print("     export OPENAI_API_KEY=sk-xxx")
        return

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Load checkpoint
    predictions, cost_total = load_checkpoint(model)

    # Determine pending
    pending = []
    for r in test_records:
        rid = r["record_id"]
        if rid in predictions:
            continue
        img_path = r.get("cropped_image_path", "")
        if img_path and os.path.isfile(img_path):
            pending.append(r)

    if mode == "test":
        pending = pending[:5]
        print(f"\n  🧪 TEST MODE — {len(pending)} records only")
    elif limit:
        pending = pending[:limit]
        print(f"\n  ⚡ LIMITED MODE — {len(pending)} records")
    else:
        print(f"\n  🚀 FULL MODE — {len(pending)} records")

    if not pending:
        print("  ✅ All records already processed.")
        return predictions, cost_total

    # Cost estimate
    n_pending, est_cost = print_cost_estimate(pending, model)

    if mode == "full" and len(pending) > 20:
        confirm = input(f"\n  Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return predictions, cost_total

    # Run
    t_start = time.time()
    success = errors = 0

    print(f"\n  Starting at {datetime.now().strftime('%H:%M:%S')}")
    print(f"  {'─'*55}")

    for i, rec in enumerate(pending):
        rid      = rec["record_id"]
        img_path = rec["cropped_image_path"]

        try:
            # Retry up to 3 times on rate limit
            text, usage = None, None
            for attempt in range(3):
                try:
                    text, usage = call_gpt_ocr(client, img_path, model)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        wait = (attempt + 1) * 5
                        print(f"    ⏳ Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise

            actual_cost  = compute_actual_cost(usage, model)
            cost_total  += actual_cost

            # Clean up [illegible] etc
            if text.startswith("["):
                predictions[rid] = ""
            else:
                predictions[rid] = text

            success += 1

            # Progress
            gt_short   = normalize_arabic(rec.get("ground_truth", ""))[:40]
            pred_short = text[:40]
            print(f"  [{i+1:03d}/{len(pending)}] ✅ ${actual_cost:.5f} "
                  f"| tok={usage['total_tokens']}")
            print(f"    GT  : {gt_short}")
            print(f"    GPT : {pred_short}")

        except Exception as e:
            predictions[rid] = ""
            errors += 1
            print(f"  [{i+1:03d}/{len(pending)}] ❌ {str(e)[:60]}")

        # Rate limit — 4 detik antar request
        if i < len(pending) - 1:
            time.sleep(REQUEST_DELAY)

        # Auto-save
        if (i + 1) % SAVE_EVERY_N == 0:
            save_checkpoint(predictions, model, cost_total)
            elapsed = time.time() - t_start
            avg_t   = elapsed / (i + 1)
            eta     = (len(pending) - i - 1) * avg_t
            print(f"  💾 Saved. Cost=${cost_total:.4f} | "
                  f"ETA={eta/60:.1f}min")

    # Final save
    save_checkpoint(predictions, model, cost_total)

    elapsed = time.time() - t_start
    print(f"\n  {'='*55}")
    print(f"  ✅ DONE")
    print(f"  {'='*55}")
    print(f"  Processed : {success} ok, {errors} errors")
    print(f"  Time      : {elapsed:.0f}s ({elapsed/max(success,1):.1f}s/rec)")
    print(f"  Cost      : ${cost_total:.4f} total")
    print(f"  {'='*55}")

    return predictions, cost_total


# ─────────────────────────────────────────────────────────────────────────────
# REPORT & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(test_records: list, predictions: dict, model: str,
                    cost_total: float):
    """Generate metrics report and save results."""
    pred_key = f"gpt_{model.replace('-','_')}"

    # Attach predictions to records
    records_with_pred = []
    for r in test_records:
        r_copy = dict(r)
        r_copy[pred_key] = predictions.get(r["record_id"], "")
        records_with_pred.append(r_copy)

    m = compute_metrics(records_with_pred, pred_key)

    print(f"\n  {'='*60}")
    print(f"  GPT OCR RESULTS — {model}")
    print(f"  {'='*60}")
    print(f"  Records  : {m['n_total']}")
    print(f"  Coverage : {m['coverage_pct']}%")
    print(f"  CER-all  : {m['cer_incl_empty']}")
    print(f"  WER-all  : {m['wer_incl_empty']}")
    print(f"  CER-rec  : {m['cer_excl_empty']}")
    print(f"  WER-rec  : {m['wer_excl_empty']}")
    print(f"  Cost     : ${cost_total:.4f}")

    # Per-kitab
    print(f"\n  Per-kitab CER:")
    for kitab in sorted(set(r["kitab_id"] for r in test_records)):
        kr = [r for r in records_with_pred if r["kitab_id"] == kitab]
        km = compute_metrics(kr, pred_key)
        cer_str = f"{km['cer_incl_empty']:.4f}" if km['cer_incl_empty'] else "N/A"
        print(f"    {kitab}: {cer_str}")

    # Save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_out = {
        "tier"       : "3-gpt",
        "model"      : model,
        "eval_split" : "test",
        "eval_n"     : len(test_records),
        "cost_usd"   : round(cost_total, 4),
        "prompt"     : OCR_PROMPT[:100] + "...",
        "global"     : m,
        "per_kitab"  : {},
    }
    for kitab in sorted(set(r["kitab_id"] for r in test_records)):
        kr = [r for r in records_with_pred if r["kitab_id"] == kitab]
        metrics_out["per_kitab"][kitab] = compute_metrics(kr, pred_key)

    mpath = os.path.join(RESULTS_DIR, f"tier3_gpt_{model.replace('-','_')}_metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Metrics: {mpath}")

    # Save predictions
    preds_out = []
    for r in records_with_pred:
        raw = r.get(pred_key) or ""
        preds_out.append({
            "record_id"      : r["record_id"],
            "kitab_id"       : r["kitab_id"],
            "ground_truth"   : r["ground_truth"],
            "gt_normalized"  : normalize_arabic(r["ground_truth"]),
            pred_key         : raw,
            f"{pred_key}_norm": normalize_arabic(raw),
        })
    ppath = os.path.join(RESULTS_DIR, f"tier3_gpt_{model.replace('-','_')}_predictions.json")
    with open(ppath, "w") as f:
        json.dump(preds_out, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Predictions: {ppath}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CASTLE-Pegon Tier 3 GPT OCR Benchmark"
    )
    parser.add_argument("--mode", choices=["estimate", "test", "full"],
                        default="estimate")
    parser.add_argument("--model", choices=["gpt-4o", "gpt-4o-mini"],
                        default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("CASTLE-Pegon -- Tier 3 GPT OCR Benchmark")
    print("=" * 60)
    print(f"  Model : {args.model}")
    print(f"  Mode  : {args.mode}")

    # Load test records
    with open(TEST_JSON) as f:
        test_records = json.load(f)

    with open(FILTERED_JSON) as f:
        filtered = json.load(f)
    crop_lookup = {r["record_id"]: r.get("cropped_image_path", "")
                   for r in filtered}
    for r in test_records:
        if not r.get("cropped_image_path"):
            r["cropped_image_path"] = crop_lookup.get(r["record_id"], "")

    n_crop = sum(1 for r in test_records
                 if r.get("cropped_image_path")
                 and os.path.isfile(r["cropped_image_path"]))
    print(f"\n  Test records: {len(test_records)}")
    print(f"  Crop found  : {n_crop}/{len(test_records)}")

    # Load checkpoint
    predictions, cost_total = load_checkpoint(args.model)
    n_done = len(predictions)
    print(f"  Already done: {n_done}")

    # Pending
    pending = [r for r in test_records
               if r["record_id"] not in predictions
               and r.get("cropped_image_path")
               and os.path.isfile(r["cropped_image_path"])]

    if args.mode == "estimate":
        print_cost_estimate(pending, args.model)
        print(f"\n  To run: python {__file__} --mode test --model {args.model}")
        return

    # Run
    predictions, cost_total = run_gpt_ocr(
        test_records, args.model, args.mode, args.limit
    )

    # Report
    if predictions:
        generate_report(test_records, predictions, args.model, cost_total)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()