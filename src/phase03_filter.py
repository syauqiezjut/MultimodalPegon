"""
=============================================================================
CASTLE-Pegon -- Phase 03: Filter Non-Pegon Records
=============================================================================
Buang record yang ground_truth-nya tidak mengandung karakter Arab/Pegon.

Kriteria DIBUANG:
  - Pure Latin / angka / simbol (tidak ada karakter Arab sama sekali)
  - Ground truth terlalu pendek (< 2 karakter)
  - Tidak punya cropped_image_path

Kriteria DIPERTAHANKAN:
  - Mengandung minimal 1 karakter Unicode Arab (U+0600-U+06FF, dll)
  - Mixed Latin+Arab tetap dipertahankan

Input : data/processed/pegon_raw.json (sudah ada cropped_image_path)
Output: data/processed/pegon_filtered.json
        logs/phase03_removed.json

Run: python src/phase03_filter.py
=============================================================================
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PEGON_RAW_JSON, PROCESSED_DIR, LOGS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

FILTERED_JSON = os.path.join(PROCESSED_DIR, "pegon_filtered.json")

# Regex karakter Arab/Pegon
ARABIC_RE = re.compile(
    r'[\u0600-\u06FF'   # Arabic Basic
    r'\u0750-\u077F'    # Arabic Supplement
    r'\u08A0-\u08FF'    # Arabic Extended-A
    r'\uFB50-\uFDFF'    # Arabic Presentation Forms-A
    r'\uFE70-\uFEFF]'   # Arabic Presentation Forms-B
)


def classify(record: dict) -> tuple[bool, str]:
    gt   = record.get('ground_truth', '').strip()
    crop = record.get('cropped_image_path', '')

    if not gt:
        return False, 'empty_gt'

    if len(gt) < 2:
        return False, 'too_short'

    if not ARABIC_RE.search(gt):
        return False, 'no_arabic_chars'

    # Tambahkan ini:
    if not crop or not os.path.isfile(crop):
        return False, 'no_crop_image'

    return True, 'ok'


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 55)
    print("CASTLE-Pegon -- Phase 03: Filter Non-Pegon Records")
    print("=" * 55)

    with open(PEGON_RAW_JSON, 'r', encoding='utf-8') as f:
        records = json.load(f)

    print(f"  Input records: {len(records)}")

    kept    = []
    removed = []

    for rec in records:
        keep, reason = classify(rec)
        if keep:
            kept.append(rec)
        else:
            removed.append({
                'record_id'   : rec.get('record_id', ''),
                'kitab_id'    : rec.get('kitab_id', ''),
                'ground_truth': rec.get('ground_truth', ''),
                'reason'      : reason,
            })

    # Statistik
    reason_counts = Counter(r['reason'] for r in removed)

    print(f"\n  Results:")
    print(f"    Kept    : {len(kept)}")
    print(f"    Removed : {len(removed)}")
    print(f"\n  Removal reasons:")
    for reason, count in reason_counts.items():
        print(f"    {reason:<20}: {count}")

    print(f"\n  Removed records (sample):")
    for r in removed[:10]:
        gt_preview = r['ground_truth'][:40] if r['ground_truth'] else '(empty)'
        print(f"    [{r['reason']:<20}] {r['kitab_id']} | '{gt_preview}'")

    # Per-kitab breakdown
    print(f"\n  Per kitab (kept):")
    by_kitab = Counter(r['kitab_id'] for r in kept)
    for k in sorted(by_kitab):
        n_img = sum(1 for r in kept
                    if r['kitab_id'] == k
                    and r.get('cropped_image_path')
                    and os.path.isfile(r['cropped_image_path']))
        print(f"    {k}: {by_kitab[k]:>4} lines | crop ok: {n_img}")

    # Save filtered
    with open(FILTERED_JSON, 'w', encoding='utf-8') as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(FILTERED_JSON) / (1024**2)
    print(f"\n  Saved: {FILTERED_JSON} ({size_mb:.1f}MB)")

    # Save removed log
    removed_path = os.path.join(LOGS_DIR, "phase03_removed.json")
    with open(removed_path, 'w', encoding='utf-8') as f:
        json.dump(removed, f, ensure_ascii=False, indent=2)
    print(f"  Removed log: {removed_path}")

    print(f"\n  NEXT: Phase 04 -- Split train/val/test")
    print(f"        Run: python src/phase04_split.py")
    print("=" * 55)


if __name__ == "__main__":
    main()