"""
=============================================================================
CASTLE-Pegon -- Phase 07: Synthetic Data Augmentation
=============================================================================
Tujuan:
    Membuat training pairs sintetis menggunakan glyph confusion matrix
    dari Phase 06. Mengikuti pipeline Guan & Greene (ACL 2024).

Desain:
    Train : input = ground_truth + glyph noise (synthetic)
            target = ground_truth (clean)
    Val   : input = ocr_gpt4o (real GPT-4o output)
            target = ground_truth (clean)
    Test  : input = ocr_gpt4o (real GPT-4o output)
            target = ground_truth (clean)

4 Jenis error yang diinjeksi:
    1. Confusion    : karakter diganti karakter mirip (glyph matrix)
    2. Deletion     : karakter dihapus
    3. Insertion    : karakter diduplikasi
    4. Diacritic drop: harakat dihapus

3 Level severity (Tier 5a = random, Tier 5b = glyph):
    mild     : confusion=0.05, del=0.02, ins=0.01, harakat=0.10
    moderate : confusion=0.10, del=0.05, ins=0.02, harakat=0.20
    severe   : confusion=0.20, del=0.08, ins=0.03, harakat=0.40

Input:
    data/splits/train.json
    data/splits/val.json   (ocr_gpt4o sudah terisi)
    data/splits/test.json  (ocr_gpt4o sudah terisi)
    data/processed/confusion_matrix/confusion_probs.pkl

Output:
    data/augmented/
        train_glyph.json    ← glyph-based augmentation (Tier 5b)
        train_random.json   ← random augmentation (Tier 5a, untuk ablasi)
        val.json            ← input=ocr_gpt4o, target=ground_truth
        test.json           ← input=ocr_gpt4o, target=ground_truth
        aug_stats.json      ← statistik augmentasi
        aug_samples.txt     ← 20 contoh untuk inspeksi

Run: python src/phase07_augment.py
=============================================================================
"""

import os
import sys
import json
import pickle
import random
import hashlib
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPLITS_DIR, PROCESSED_DIR, BASE_DIR

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

CONFUSION_PKL  = os.path.join(
    PROCESSED_DIR, "confusion_matrix", "confusion_probs.pkl"
)
AUG_DIR        = os.path.join(BASE_DIR, "data", "augmented")
TRAIN_JSON     = os.path.join(SPLITS_DIR, "train.json")
VAL_JSON       = os.path.join(SPLITS_DIR, "val.json")
TEST_JSON      = os.path.join(SPLITS_DIR, "test.json")

RANDOM_SEED    = 42
MIN_TEXT_LEN   = 3

# 3 severity levels — 1 original + 3 augmented = 4x total
AUG_CONFIGS = [
    # (name,       confuse, delete, insert, harakat_drop)
    ("mild",        0.05,   0.02,   0.01,   0.10),
    ("moderate",    0.10,   0.05,   0.02,   0.20),
    ("severe",      0.20,   0.08,   0.03,   0.40),
]

# ─────────────────────────────────────────────────────────────────────────────
# KARAKTER PEGON
# ─────────────────────────────────────────────────────────────────────────────

PEGON_CHARS = {
    'alif': '\u0627', 'ba': '\u0628', 'ta': '\u062a', 'tsa': '\u062b',
    'jim': '\u062c', 'ha_kecil': '\u062d', 'kha': '\u062e', 'dal': '\u062f',
    'dzal': '\u0630', 'ra': '\u0631', 'zain': '\u0632', 'sin': '\u0633',
    'syin': '\u0634', 'shad': '\u0635', 'dhad': '\u0636', 'tha': '\u0637',
    'zha': '\u0638', 'ain': '\u0639', 'ghain': '\u063a', 'fa': '\u0641',
    'qaf': '\u0642', 'kaf': '\u0643', 'lam': '\u0644', 'mim': '\u0645',
    'nun': '\u0646', 'ha_besar': '\u0647', 'waw': '\u0648', 'ya': '\u064a',
    'ta_marbuta': '\u0629', 'alif_maqsura': '\u0649', 'hamza': '\u0621',
    'alif_hamza': '\u0623', 'alif_madda': '\u0622',
    'pa': '\u06a4', 'nga': '\u06a0', 'ga': '\u06ae', 'ca': '\u0684',
    'nya': '\u06bd', 'nya2': '\u06d1',
}

CHAR_TO_NAME  = {v: k for k, v in PEGON_CHARS.items()}
NAME_TO_CHAR  = PEGON_CHARS
ARABIC_CHARS  = set(chr(cp) for cp in range(0x0600, 0x0700))
DIACRITICS    = set('\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652'
                    '\u0670\u0653')

# Fallback probs jika PKL tidak tersedia
FALLBACK_PROBS = {
    'ba': {'ta': 0.35, 'tsa': 0.25, 'nun': 0.20, 'ya': 0.20},
    'ta': {'ba': 0.35, 'tsa': 0.30, 'nun': 0.20, 'ya': 0.15},
    'tsa': {'ba': 0.35, 'ta': 0.35, 'nun': 0.15, 'ya': 0.15},
    'nun': {'ba': 0.35, 'ta': 0.30, 'ya': 0.20, 'tsa': 0.15},
    'ha_kecil': {'jim': 0.55, 'kha': 0.45},
    'jim': {'ha_kecil': 0.55, 'kha': 0.45},
    'kha': {'ha_kecil': 0.55, 'jim': 0.45},
    'dal': {'dzal': 0.60, 'ra': 0.25, 'zain': 0.15},
    'ra': {'zain': 0.55, 'dal': 0.25, 'dzal': 0.20},
    'sin': {'syin': 0.70, 'shad': 0.30},
    'ain': {'ghain': 0.70, 'tha': 0.30},
    'fa': {'qaf': 0.60, 'pa': 0.40},
    'qaf': {'fa': 0.55, 'pa': 0.45},
    'pa': {'fa': 0.50, 'qaf': 0.50},
    'nga': {'ain': 0.55, 'ghain': 0.45},
    'ga': {'kaf': 0.70, 'qaf': 0.30},
    'ca': {'jim': 0.60, 'kha': 0.40},
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CONFUSION PROBS
# ─────────────────────────────────────────────────────────────────────────────

def load_confusion_probs():
    if os.path.isfile(CONFUSION_PKL):
        with open(CONFUSION_PKL, 'rb') as f:
            data = pickle.load(f)
        probs = data.get('probs', {})
        c2n   = data.get('char_to_name', CHAR_TO_NAME)
        n2c   = data.get('name_to_char', NAME_TO_CHAR)
        print(f"  ✅ Glyph confusion probs loaded: {len(probs)} chars")
        return probs, c2n, n2c, "glyph"
    else:
        print(f"  ⚠️  confusion_probs.pkl tidak ditemukan")
        print(f"     Fallback: rule-based confusion")
        return FALLBACK_PROBS, CHAR_TO_NAME, NAME_TO_CHAR, "fallback"


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISASI
# ─────────────────────────────────────────────────────────────────────────────

def normalize_arabic(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\u0640", "")
    return " ".join(text.split()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# ERROR INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def inject_errors(text, probs, char_to_name, name_to_char,
                  confusion_rate=0.10, deletion_rate=0.05,
                  insertion_rate=0.02, diacritic_drop_rate=0.20,
                  rng=None):
    """
    Injeksi error OCR-realistis ke teks Pegon bersih.
    Urutan per karakter: diacritic drop → delete → confuse → insert
    """
    if rng is None:
        rng = random.Random()

    result = []
    errors = []

    for pos, char in enumerate(text):
        is_diac   = char in DIACRITICS
        is_arabic = char in ARABIC_CHARS and not is_diac
        name      = char_to_name.get(char)

        # Rule 1: Drop diacritic
        if is_diac:
            if rng.random() < diacritic_drop_rate:
                errors.append({'method': 'diacritic_drop',
                               'orig': char, 'replaced': ''})
                continue
            result.append(char)
            continue

        # Rule 2: Delete
        if is_arabic and rng.random() < deletion_rate:
            errors.append({'method': 'deletion',
                           'orig': char, 'replaced': ''})
            continue

        # Rule 3: Confusion substitution
        if name and name in probs and rng.random() < confusion_rate:
            targets = list(probs[name].keys())
            weights = list(probs[name].values())
            if targets and sum(weights) > 1e-10:
                chosen   = rng.choices(targets, weights=weights, k=1)[0]
                replaced = name_to_char.get(chosen, char)
                errors.append({'method': 'confusion',
                               'orig': char, 'replaced': replaced})
                result.append(replaced)
                # Rule 4: Insertion setelah substitusi
                if rng.random() < insertion_rate:
                    result.append(replaced)
                    errors.append({'method': 'insertion',
                                   'orig': '', 'replaced': replaced})
                continue

        result.append(char)

    return ''.join(result), errors


def inject_random(text, rng=None,
                  sub_rate=0.10, del_rate=0.05, ins_rate=0.02,
                  diac_rate=0.20):
    """
    Random injection (Tier 5a baseline) — tanpa glyph similarity.
    Substitusi dengan karakter Arab acak dari PEGON_CHARS.
    """
    if rng is None:
        rng = random.Random()

    all_chars = list(PEGON_CHARS.values())
    result    = []
    errors    = []

    for pos, char in enumerate(text):
        is_diac   = char in DIACRITICS
        is_arabic = char in ARABIC_CHARS and not is_diac

        if is_diac:
            if rng.random() < diac_rate:
                errors.append({'method': 'diacritic_drop',
                               'orig': char, 'replaced': ''})
                continue
            result.append(char)
            continue

        if is_arabic and rng.random() < del_rate:
            errors.append({'method': 'deletion',
                           'orig': char, 'replaced': ''})
            continue

        if is_arabic and rng.random() < sub_rate:
            replaced = rng.choice(all_chars)
            errors.append({'method': 'random_sub',
                           'orig': char, 'replaced': replaced})
            result.append(replaced)
            if rng.random() < ins_rate:
                result.append(replaced)
            continue

        result.append(char)

    return ''.join(result), errors


def compute_cer(ref, hyp):
    if not ref:
        return 0.0
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] if r[i-1] == h[j-1] \
                else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m] / len(r)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_train(train_records, probs, char_to_name, name_to_char):
    """
    Buat dua versi augmented training data:
    - glyph  : menggunakan confusion matrix (Tier 5b)
    - random : substitusi acak (Tier 5a, untuk ablasi)
    """
    glyph_pairs  = []
    random_pairs = []
    stats        = {
        'total_original'  : 0,
        'total_augmented' : 0,
        'skipped'         : 0,
        'errors_per_type' : Counter(),
        'cer_per_config'  : defaultdict(list),
    }

    for rec in train_records:
        gt = rec.get('ground_truth', '').strip()
        if len(gt) < MIN_TEXT_LEN:
            stats['skipped'] += 1
            continue

        # Original pair (clean → clean)
        base_pair = {
            'id'       : f"{rec['record_id']}_original",
            'source_id': rec['record_id'],
            'input'    : rec.get('ocr_gpt4o', gt),
            'target'   : gt,
            'aug_type' : 'original',
            'kitab_id' : rec['kitab_id'],
            'language' : rec.get('language', ''),
            'split'    : 'train',
        }
        glyph_pairs.append(base_pair)
        random_pairs.append(base_pair.copy())
        stats['total_original'] += 1

        # Augmented pairs per config
        for aug_type, conf_r, del_r, ins_r, dia_r in AUG_CONFIGS:
            # Deterministic seed per record+type
            local_seed = int(hashlib.md5(
                f"{rec['record_id']}_{aug_type}".encode()
            ).hexdigest()[:8], 16)
            local_rng = random.Random(local_seed)

            # ── Glyph augmentation ──
            noisy_g, errs_g = inject_errors(
                gt, probs, char_to_name, name_to_char,
                confusion_rate      = conf_r,
                deletion_rate       = del_r,
                insertion_rate      = ins_r,
                diacritic_drop_rate = dia_r,
                rng                 = local_rng,
            )
            if noisy_g != gt:
                cer = compute_cer(gt, noisy_g)
                stats['cer_per_config'][f"glyph_{aug_type}"].append(cer)
                for e in errs_g:
                    stats['errors_per_type'][e['method']] += 1
                glyph_pairs.append({
                    'id'       : f"{rec['record_id']}_glyph_{aug_type}",
                    'source_id': rec['record_id'],
                    'input'    : noisy_g,
                    'target'   : gt,
                    'aug_type' : f"glyph_{aug_type}",
                    'kitab_id' : rec['kitab_id'],
                    'language' : rec.get('language', ''),
                    'split'    : 'train',
                    'injected_cer': round(cer, 4),
                })
                stats['total_augmented'] += 1

            # ── Random augmentation ──
            local_rng2 = random.Random(local_seed + 1)
            noisy_r, _ = inject_random(
                gt, rng=local_rng2,
                sub_rate=conf_r, del_rate=del_r,
                ins_rate=ins_r, diac_rate=dia_r,
            )
            if noisy_r != gt:
                random_pairs.append({
                    'id'       : f"{rec['record_id']}_random_{aug_type}",
                    'source_id': rec['record_id'],
                    'input'    : noisy_r,
                    'target'   : gt,
                    'aug_type' : f"random_{aug_type}",
                    'kitab_id' : rec['kitab_id'],
                    'language' : rec.get('language', ''),
                    'split'    : 'train',
                    'injected_cer': round(compute_cer(gt, noisy_r), 4),
                })

    return glyph_pairs, random_pairs, stats


def make_eval_pairs(records, split_name, ocr_field='ocr_gpt4o'):
    """
    Buat eval pairs dengan input dari real OCR output.
    input  = ocr_gpt4o (real GPT-4o output)
    target = ground_truth (clean)
    """
    pairs   = []
    n_empty = 0

    for r in records:
        gt   = r.get('ground_truth', '').strip()
        ocr  = r.get(ocr_field, '').strip()

        if not gt:
            continue
        if not ocr:
            n_empty += 1
            ocr = gt  # fallback ke clean jika OCR kosong

        pairs.append({
            'id'       : r['record_id'],
            'source_id': r['record_id'],
            'input'    : ocr,
            'target'   : gt,
            'aug_type' : 'real_ocr',
            'kitab_id' : r.get('kitab_id', ''),
            'language' : r.get('language', ''),
            'split'    : split_name,
        })

    if n_empty > 0:
        print(f"  ⚠️  {n_empty} records in {split_name} have empty OCR → used GT as fallback")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# SAVE & REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_json(records, path, name):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    size = os.path.getsize(path) / (1024**2)
    print(f"  ✅ {name}: {len(records)} pairs ({size:.1f}MB) → {path}")


def save_samples(glyph_pairs, output_path, n=20):
    by_type = defaultdict(list)
    for p in glyph_pairs:
        if p['aug_type'] != 'original':
            by_type[p['aug_type']].append(p)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("CASTLE-Pegon — Augmentation Samples (Glyph-based)\n")
        f.write("=" * 60 + "\n\n")
        shown = 0
        for aug_type, pairs in sorted(by_type.items()):
            f.write(f"── {aug_type.upper()} ──\n")
            for p in pairs[:2]:
                if shown >= n:
                    break
                f.write(f"  Target   : {p['target']}\n")
                f.write(f"  Input    : {p['input']}\n")
                f.write(f"  CER      : {p.get('injected_cer', '?')}\n\n")
                shown += 1
    print(f"  ✅ Samples: {output_path}")


def print_report(glyph, random_p, val, test, stats, method):
    print(f"\n{'='*60}")
    print("AUGMENTATION REPORT")
    print(f"{'='*60}")
    print(f"  Method: {method}")
    print(f"\n  Training pairs:")
    print(f"    Glyph augmented : {len(glyph)}")
    print(f"    Random augmented: {len(random_p)}")
    print(f"    Original records: {stats['total_original']}")
    print(f"    Augmented added : {stats['total_augmented']}")
    print(f"    Skipped (short) : {stats['skipped']}")

    if stats['total_original'] > 0:
        ratio = len(glyph) / stats['total_original']
        print(f"    Multiplier      : {ratio:.1f}x")

    print(f"\n  Eval pairs:")
    print(f"    Val  : {len(val)} (input=ocr_gpt4o)")
    print(f"    Test : {len(test)} (input=ocr_gpt4o)")

    print(f"\n  Error distribution (glyph):")
    total = sum(stats['errors_per_type'].values())
    for method_name, count in sorted(
        stats['errors_per_type'].items(), key=lambda x: -x[1]
    ):
        pct = count / total * 100 if total > 0 else 0
        print(f"    {method_name:<20}: {count:>5} ({pct:.1f}%)")

    print(f"\n  Avg injected CER per config:")
    for cfg, cers in sorted(stats['cer_per_config'].items()):
        if cers:
            avg = sum(cers) / len(cers)
            print(f"    {cfg:<25}: {avg:.4f} ({avg*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(AUG_DIR, exist_ok=True)

    print("=" * 60)
    print("CASTLE-Pegon -- Phase 07: Synthetic Data Augmentation")
    print("=" * 60)
    print(f"  Output: {AUG_DIR}")

    # Load confusion probs
    print("\nLoading confusion probabilities...")
    probs, char_to_name, name_to_char, method = load_confusion_probs()

    # Load splits
    print("\nLoading splits...")
    with open(TRAIN_JSON) as f:
        train_records = json.load(f)
    with open(VAL_JSON) as f:
        val_records = json.load(f)
    with open(TEST_JSON) as f:
        test_records = json.load(f)

    print(f"  Train: {len(train_records)}")
    print(f"  Val  : {len(val_records)}")
    print(f"  Test : {len(test_records)}")

    # Cek ocr_gpt4o di val dan test
    val_has_gpt  = sum(1 for r in val_records  if r.get('ocr_gpt4o'))
    test_has_gpt = sum(1 for r in test_records if r.get('ocr_gpt4o'))
    print(f"\n  Val  ocr_gpt4o: {val_has_gpt}/{len(val_records)}")
    print(f"  Test ocr_gpt4o: {test_has_gpt}/{len(test_records)}")

    # Augment training data
    print(f"\nAugmenting training data ({len(AUG_CONFIGS)} configs)...")
    print(f"  Configs: {[c[0] for c in AUG_CONFIGS]}")
    glyph_pairs, random_pairs, stats = augment_train(
        train_records, probs, char_to_name, name_to_char
    )

    # Shuffle training pairs
    random.Random(RANDOM_SEED).shuffle(glyph_pairs)
    random.Random(RANDOM_SEED).shuffle(random_pairs)

    # Make eval pairs (input = ocr_gpt4o)
    val_pairs  = make_eval_pairs(val_records,  'val',  'ocr_gpt4o')
    test_pairs = make_eval_pairs(test_records, 'test', 'ocr_gpt4o')

    # Save
    print(f"\nSaving files...")
    save_json(glyph_pairs,  os.path.join(AUG_DIR, "train_glyph.json"),  "train_glyph")
    save_json(random_pairs, os.path.join(AUG_DIR, "train_random.json"), "train_random")
    save_json(val_pairs,    os.path.join(AUG_DIR, "val.json"),          "val")
    save_json(test_pairs,   os.path.join(AUG_DIR, "test.json"),         "test")

    # Stats
    stats_out = {
        'method'          : method,
        'aug_configs'     : [
            {'name': c[0], 'confusion_rate': c[1], 'deletion_rate': c[2],
             'insertion_rate': c[3], 'diacritic_drop_rate': c[4]}
            for c in AUG_CONFIGS
        ],
        'sizes': {
            'train_glyph' : len(glyph_pairs),
            'train_random': len(random_pairs),
            'val'         : len(val_pairs),
            'test'        : len(test_pairs),
        },
        'stats': {
            'total_original'  : stats['total_original'],
            'total_augmented' : stats['total_augmented'],
            'skipped'         : stats['skipped'],
            'errors_per_type' : dict(stats['errors_per_type']),
            'cer_per_config'  : {
                k: {'mean': round(sum(v)/len(v), 4), 'n': len(v)}
                for k, v in stats['cer_per_config'].items()
            },
        },
        'random_seed': RANDOM_SEED,
    }
    stats_path = os.path.join(AUG_DIR, "aug_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)
    print(f"  ✅ aug_stats.json")

    # Samples
    save_samples(glyph_pairs, os.path.join(AUG_DIR, "aug_samples.txt"))

    # Report
    print_report(glyph_pairs, random_pairs, val_pairs, test_pairs,
                 stats, method)

    print(f"\n{'='*60}")
    print("PHASE 07 COMPLETE")
    print(f"{'='*60}")
    print(f"  train_glyph.json  ← untuk training Tier 5b (ByT5 + glyph aug)")
    print(f"  train_random.json ← untuk training Tier 5a (ByT5 + random aug)")
    print(f"  val.json          ← input=GPT-4o output")
    print(f"  test.json         ← input=GPT-4o output")
    print(f"\n  NEXT: python src/phase08_byt5.py")
    print("=" * 60)


if __name__ == "__main__":
    main()