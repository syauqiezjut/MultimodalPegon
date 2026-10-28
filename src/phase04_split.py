"""
=============================================================================
CASTLE-Pegon -- Phase 04: Split Train/Val/Test
=============================================================================
Strategi split: per kitab (bukan random global)
  - Hindari data leakage: halaman dari kitab sama tidak tersebar di split berbeda
  - Split per halaman: 70% train / 15% val / 15% test
  - Seed fixed untuk reproducibility

Input : data/processed/pegon_filtered.json
Output: data/splits/train.json
        data/splits/val.json
        data/splits/test.json
        data/splits/split_stats.json

Run: python src/phase04_split.py
=============================================================================
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, SPLITS_DIR, LOGS_DIR, SPLIT_RATIO, SPLIT_SEED

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

FILTERED_JSON = os.path.join(PROCESSED_DIR, "pegon_filtered.json")


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def split_kitab(records: list, ratio: dict, seed: int) -> dict:
    """
    Split records per halaman untuk satu kitab.
    Semua baris dari halaman yang sama masuk ke split yang sama.

    Returns: {'train': [...], 'val': [...], 'test': [...]}
    """
    # Group by page_number
    pages = defaultdict(list)
    for rec in records:
        pages[rec['page_number']].append(rec)

    page_nums = sorted(pages.keys())
    random.seed(seed)
    random.shuffle(page_nums)

    n         = len(page_nums)
    n_train   = max(1, int(n * ratio['train']))
    n_val     = max(1, int(n * ratio['val']))
    # test = sisa

    train_pages = page_nums[:n_train]
    val_pages   = page_nums[n_train:n_train + n_val]
    test_pages  = page_nums[n_train + n_val:]

    result = {'train': [], 'val': [], 'test': []}
    for pnum in train_pages:
        result['train'].extend(pages[pnum])
    for pnum in val_pages:
        result['val'].extend(pages[pnum])
    for pnum in test_pages:
        result['test'].extend(pages[pnum])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 55)
    print("CASTLE-Pegon -- Phase 04: Split Train/Val/Test")
    print("=" * 55)
    print(f"  Ratio : train={SPLIT_RATIO['train']} / "
          f"val={SPLIT_RATIO['val']} / "
          f"test={SPLIT_RATIO['test']}")
    print(f"  Seed  : {SPLIT_SEED}")
    print(f"  Strategy: per-page split (no leakage)")

    with open(FILTERED_JSON, 'r', encoding='utf-8') as f:
        records = json.load(f)

    print(f"\n  Input records: {len(records)}")

    # Group by kitab
    by_kitab = defaultdict(list)
    for rec in records:
        by_kitab[rec['kitab_id']].append(rec)

    # Split per kitab
    all_splits = {'train': [], 'val': [], 'test': []}
    split_stats = {}

    print(f"\n  Per-kitab split:")
    for kitab_id in sorted(by_kitab.keys()):
        kitab_records = by_kitab[kitab_id]
        splits        = split_kitab(kitab_records, SPLIT_RATIO, SPLIT_SEED)

        for split_name, recs in splits.items():
            # Tambah field split ke setiap record
            for r in recs:
                r['split'] = split_name
            all_splits[split_name].extend(recs)

        # Hitung pages per split
        train_pages = len(set(r['page_number'] for r in splits['train']))
        val_pages   = len(set(r['page_number'] for r in splits['val']))
        test_pages  = len(set(r['page_number'] for r in splits['test']))

        split_stats[kitab_id] = {
            'total'      : len(kitab_records),
            'train'      : len(splits['train']),
            'val'        : len(splits['val']),
            'test'       : len(splits['test']),
            'train_pages': train_pages,
            'val_pages'  : val_pages,
            'test_pages' : test_pages,
        }

        print(f"    {kitab_id}: "
              f"train={len(splits['train'])}({train_pages}p) | "
              f"val={len(splits['val'])}({val_pages}p) | "
              f"test={len(splits['test'])}({test_pages}p)")

    # Summary
    print(f"\n  Global split:")
    for split_name in ['train', 'val', 'test']:
        recs    = all_splits[split_name]
        n_crop  = sum(1 for r in recs
                      if r.get('cropped_image_path')
                      and os.path.isfile(r['cropped_image_path']))
        by_lang = Counter(r['language'] for r in recs)
        lang_str = ', '.join(f"{l}:{n}" for l, n in by_lang.items())
        print(f"    {split_name:<6}: {len(recs):>4} records | "
              f"crop ok: {n_crop} | {lang_str}")

    # Save splits
    for split_name in ['train', 'val', 'test']:
        out_path = os.path.join(SPLITS_DIR, f"{split_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_splits[split_name], f, ensure_ascii=False, indent=2)
        size_mb = os.path.getsize(out_path) / (1024**2)
        print(f"    Saved: {out_path} ({size_mb:.1f}MB)")

    # Save stats
    stats_out = {
        'total'     : len(records),
        'train'     : len(all_splits['train']),
        'val'       : len(all_splits['val']),
        'test'      : len(all_splits['test']),
        'ratio'     : SPLIT_RATIO,
        'seed'      : SPLIT_SEED,
        'per_kitab' : split_stats,
    }
    stats_path = os.path.join(SPLITS_DIR, "split_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)

    print(f"\n  Stats: {stats_path}")
    print(f"\n  NEXT: Phase 05 -- Run OCR engines")
    print(f"        Run: python src/phase05_ocr.py")
    print("=" * 55)


if __name__ == "__main__":
    main()