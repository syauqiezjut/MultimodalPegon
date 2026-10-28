"""
=============================================================================
CASTLE-Pegon -- Phase 05: Tier 1 OCR Benchmarking
=============================================================================
Engine  : Tesseract 5 (ara) + EasyOCR (ar)
Tujuan  : Establish lower bound — tunjukkan classical pipeline gagal pada Pegon

Fixes v2:
  1. CER/WER tracking n_empty terpisah, report excluding empty juga
  2. EasyOCR paragraph=False untuk single line images
  3. EASY_GPU auto-detect via torch.cuda.is_available()
  4. normalize_arabic() sebelum CER/WER computation
  5. compute_metrics report both including dan excluding empty predictions

Input : data/processed/pegon_filtered.json
Output: data/processed/pegon_filtered.json (updated in-place)
        results/ocr/tier1_metrics.json
        results/ocr/tier1_predictions_test.json

Run: python src/phase05_ocr_tier1.py
=============================================================================
"""

import os
import sys
import json
import time
import subprocess
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, LOGS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

FILTERED_JSON = os.path.join(PROCESSED_DIR, "pegon_filtered.json")
RESULTS_DIR   = os.path.join(
    str(Path(__file__).parent.parent), "results", "ocr"
)

# Tesseract
TESS_LANG = "ara"
TESS_PSM  = "7"    # single text line
TESS_OEM  = "1"    # LSTM only

# EasyOCR
EASY_LANGS = ["ar"]

# Auto-detect GPU (fix poin 3)
try:
    import torch
    EASY_GPU = torch.cuda.is_available()
except ImportError:
    EASY_GPU = False

print(f"GPU available: {EASY_GPU}")

REPORT_EVERY = 100


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION (fix poin 4)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """
    Normalisasi teks Arab sebelum CER/WER computation.
    Menghindari inflate CER karena perbedaan Unicode encoding.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "")   # ZWNJ
    text = text.replace("\u200d", "")   # ZWJ
    text = text.replace("\u0640", "")   # kashida/tatweel
    text = " ".join(text.split())       # normalize whitespace
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_tesseract() -> bool:
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True, text=True, timeout=10
        )
        langs = result.stdout + result.stderr
        if TESS_LANG in langs:
            print(f"  ✅ Tesseract: Arabic pack tersedia")
            return True
        else:
            print(f"  ⚠️  Tesseract: Arabic pack BELUM tersedia")
            print(f"     Run: brew install tesseract-lang")
            return False
    except Exception as e:
        print(f"  ❌ Tesseract error: {e}")
        return False


def check_easyocr() -> bool:
    try:
        import easyocr
        print(f"  ✅ EasyOCR: tersedia (v{easyocr.__version__})")
        return True
    except ImportError:
        print(f"  ❌ EasyOCR tidak ditemukan")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# OCR FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_tesseract(image_path: str) -> str:
    try:
        cmd = [
            "tesseract", image_path, "stdout",
            "-l", TESS_LANG,
            "--psm", TESS_PSM,
            "--oem", TESS_OEM,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30, encoding="utf-8"
        )
        return result.stdout.strip()
    except Exception:
        return ""


def init_easyocr():
    import easyocr
    print(f"  Initializing EasyOCR (langs={EASY_LANGS}, gpu={EASY_GPU})...")
    reader = easyocr.Reader(EASY_LANGS, gpu=EASY_GPU, verbose=False)
    print(f"  ✅ EasyOCR ready")
    return reader


def run_easyocr(reader, image_path: str) -> str:
    try:
        results = reader.readtext(
            image_path,
            detail    = 0,      # text only
            paragraph = False,  # fix poin 2: False untuk single line images
        )
        text = " ".join(str(r) for r in results).strip()
        return text
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# METRICS (fix poin 1 — track empty terpisah)
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


def compute_metrics(records: list, ocr_field: str) -> dict:
    """
    Hitung CER/WER dengan dua versi:
    - including_empty : semua records (empty pred = CER 1.0)
    - excluding_empty : hanya records yang punya prediksi non-empty

    fix poin 1: track n_empty terpisah, report keduanya.
    """
    cer_all, wer_all         = [], []   # termasuk empty
    cer_nonempty, wer_nonempty = [], [] # hanya non-empty
    n_empty   = 0
    n_skipped = 0  # record tanpa GT

    for r in records:
        pred_raw = r.get(ocr_field) or ""
        gt_raw   = r.get("ground_truth", "")

        if not gt_raw:
            n_skipped += 1
            continue

        # Normalize keduanya sebelum hitung (fix poin 4)
        pred = normalize_arabic(pred_raw)
        gt   = normalize_arabic(gt_raw)

        cer = edit_distance(pred, gt) / len(gt) if gt else 0.0
        wer_p, wer_g = pred.split(), gt.split()
        wer = edit_distance(wer_p, wer_g) / len(wer_g) if wer_g else 0.0

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
        "n_total"          : len(records) - n_skipped,
        "n_empty_pred"     : n_empty,
        "n_nonempty_pred"  : len(cer_nonempty),
        "coverage_pct"     : round(len(cer_nonempty) / max(len(cer_all), 1) * 100, 1),
        # Including empty (empty pred = CER 1.0)
        "cer_incl_empty"   : avg(cer_all),
        "wer_incl_empty"   : avg(wer_all),
        # Excluding empty (lebih fair untuk menilai kualitas actual recognition)
        "cer_excl_empty"   : avg(cer_nonempty),
        "wer_excl_empty"   : avg(wer_nonempty),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_engine(records: list,
               engine_name: str,
               ocr_field: str,
               ocr_fn) -> list:
    """
    Jalankan satu OCR engine pada semua records.
    Resume-friendly: skip jika sudah ada hasil.
    """
    print(f"\n{'─'*55}")
    print(f"Running {engine_name}...")
    print(f"{'─'*55}")

    to_process = [
        r for r in records
        if r.get(ocr_field) is None
        and r.get('cropped_image_path')
        and os.path.isfile(r['cropped_image_path'])
    ]
    already = sum(1 for r in records if r.get(ocr_field) is not None)

    print(f"  To process  : {len(to_process)}")
    print(f"  Already done: {already} (skip)")

    if not to_process:
        print(f"  ✅ All records already processed")
        return records

    t_start = time.time()
    success = 0
    failed  = 0

    for i, rec in enumerate(to_process):
        img_path = rec['cropped_image_path']
        try:
            text = ocr_fn(img_path)
            rec[ocr_field] = text
            success += 1
        except Exception as e:
            rec[ocr_field] = ""
            failed += 1

        if (i + 1) % REPORT_EVERY == 0:
            elapsed   = time.time() - t_start
            per_img   = elapsed / (i + 1)
            remaining = (len(to_process) - i - 1) * per_img
            print(f"  [{i+1}/{len(to_process)}] "
                  f"ok={success} fail={failed} | "
                  f"{elapsed:.0f}s elapsed | "
                  f"ETA {remaining:.0f}s")

    elapsed = time.time() - t_start
    print(f"\n  ✅ Done: {success} ok, {failed} failed | {elapsed:.1f}s")
    print(f"  Avg: {elapsed/max(len(to_process),1)*1000:.0f}ms per image")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(records: list, engines: list):
    # Filter test split dulu
    test_records = [r for r in records if r.get('split') == 'test']

    print(f"\n{'='*65}")
    print("TIER 1 BENCHMARK RESULTS")
    print(f"{'='*65}")
    print(f"\n  Evaluating on: test split ({len(test_records)} records)")
    print(f"  Note: CER dihitung SETELAH normalisasi Unicode (NFC, ZWNJ, kashida)")

    all_metrics = {}
    for engine_name, ocr_field in engines:
        m = compute_metrics(test_records, ocr_field)  # ← test_records
        all_metrics[engine_name] = m

    # Global table
    print(f"\n  Global (test split, {len(test_records)} records):")  # ← test_records
    print(f"  {'Engine':<20} {'Coverage':>9} {'CER(all)':>9} "
          f"{'WER(all)':>9} {'CER(non∅)':>10} {'WER(non∅)':>10}")
    print(f"  {'─'*69}")
    for engine_name, _ in engines:
        m = all_metrics[engine_name]
        cov = f"{m['coverage_pct']:.1f}%"
        ci  = f"{m['cer_incl_empty']:.4f}" if m['cer_incl_empty'] is not None else "N/A"
        wi  = f"{m['wer_incl_empty']:.4f}" if m['wer_incl_empty'] is not None else "N/A"
        ce  = f"{m['cer_excl_empty']:.4f}" if m['cer_excl_empty'] is not None else "N/A"
        we  = f"{m['wer_excl_empty']:.4f}" if m['wer_excl_empty'] is not None else "N/A"
        print(f"  {engine_name:<20} {cov:>9} {ci:>9} {wi:>9} {ce:>10} {we:>10}")

    # Per-kitab — pakai test_records
    print(f"\n  Per-kitab CER (test split, including empty):")
    print(f"  {'Kitab':<10}", end="")
    for engine_name, _ in engines:
        print(f" {engine_name[:14]:>14}", end="")
    print()
    print(f"  {'─'*55}")
    for kitab in sorted(set(r['kitab_id'] for r in test_records)):  # ← test_records
        kitab_recs = [r for r in test_records if r['kitab_id'] == kitab]  # ← test_records
        print(f"  {kitab:<10}", end="")
        for _, ocr_field in engines:
            m   = compute_metrics(kitab_recs, ocr_field)
            val = f"{m['cer_incl_empty']:.4f}" \
                  if m['cer_incl_empty'] is not None else "N/A"
            print(f" {val:>14}", end="")
        print()

    # Sample predictions — dari test_records
    print(f"\n  Sample predictions (3 test records):")
    for rec in test_records[:3]:  # ← test_records
        gt = normalize_arabic(rec.get('ground_truth', ''))[:45]
        print(f"\n  GT : {gt}")
        for engine_name, ocr_field in engines:
            pred = normalize_arabic(rec.get(ocr_field) or "")[:45]
            pred_display = pred if pred else "(empty)"
            print(f"  {engine_name:<20}: {pred_display}")

    return all_metrics


def save_results(records: list, metrics: dict, engines: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    test_records = [r for r in records if r.get('split') == 'test']  # ← tambah ini

    out = {"tier": 1, "description": "Traditional OCR lower bound",
           "eval_split": "test", "eval_n": len(test_records), "engines": {}}

    for engine_name, ocr_field in engines:
        m = metrics.get(engine_name, {})
        per_kitab = {}
        for kitab in sorted(set(r['kitab_id'] for r in test_records)):  # ← test_records
            kr = [r for r in test_records if r['kitab_id'] == kitab]   # ← test_records
            per_kitab[kitab] = compute_metrics(kr, ocr_field)
        out["engines"][engine_name] = {
            "ocr_field" : ocr_field,
            "eval_split": "test",
            "global"    : m,
            "per_kitab" : per_kitab,
        }

    metrics_path = os.path.join(RESULTS_DIR, "tier1_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Metrics saved: {metrics_path}")

    # Test predictions — sudah benar, tidak perlu diubah
    preds = []
    for r in test_records:
        entry = {
            'record_id'    : r['record_id'],
            'kitab_id'     : r['kitab_id'],
            'ground_truth' : r['ground_truth'],
            'gt_normalized': normalize_arabic(r['ground_truth']),
        }
        for _, ocr_field in engines:
            raw = r.get(ocr_field) or ""
            entry[ocr_field]                 = raw
            entry[f"{ocr_field}_normalized"] = normalize_arabic(raw)
        preds.append(entry)

    preds_path = os.path.join(RESULTS_DIR, "tier1_predictions_test.json")
    with open(preds_path, 'w', encoding='utf-8') as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Test predictions: {preds_path} ({len(preds)} records)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 55)
    print("CASTLE-Pegon -- Phase 05: Tier 1 OCR Benchmark")
    print("=" * 55)
    print("  Engines : Tesseract (ara) + EasyOCR (ar)")
    print("  Tujuan  : Establish lower bound for Pegon OCR")
    print("  Fixes   : normalize_arabic, paragraph=False,")
    print("            GPU auto-detect, empty tracking")

    print(f"\nChecking engine availability...")
    tess_ok = check_tesseract()
    easy_ok = check_easyocr()

    if not tess_ok and not easy_ok:
        print("\n❌ Tidak ada engine tersedia.")
        return

    with open(FILTERED_JSON, 'r', encoding='utf-8') as f:
        records = json.load(f)
    print(f"\n  Total records: {len(records)}")

    engines_run = []

    # ── Tesseract ──
    if tess_ok:
        records = run_engine(
            records, "Tesseract", "ocr_tesseract", run_tesseract
        )
        engines_run.append(("Tesseract", "ocr_tesseract"))
        with open(FILTERED_JSON, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  💾 Checkpoint saved (Tesseract)")
    else:
        print(f"\n  ⏭️  Tesseract skipped — jalankan ulang setelah:")
        print(f"     brew install tesseract-lang")

    # ── EasyOCR ──
    if easy_ok:
        reader  = init_easyocr()
        records = run_engine(
            records, "EasyOCR", "ocr_easyocr",
            lambda p: run_easyocr(reader, p)
        )
        engines_run.append(("EasyOCR", "ocr_easyocr"))
        with open(FILTERED_JSON, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  💾 Checkpoint saved (EasyOCR)")

    if not engines_run:
        print("\n⚠️  Tidak ada engine yang berhasil dijalankan.")
        return

    metrics = print_report(records, engines_run)
    save_results(records, metrics, engines_run)

    print(f"\n{'='*55}")
    print("PHASE 05 TIER 1 COMPLETE")
    print(f"{'='*55}")
    print(f"  Updated: {FILTERED_JSON}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"\n  NEXT: Phase 05 Tier 2 -- TrOCR variants")
    print(f"        Run: python src/phase05_ocr_tier2.py")
    print("=" * 55)


if __name__ == "__main__":
    main()