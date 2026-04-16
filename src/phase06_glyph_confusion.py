"""
=============================================================================
CASTLE-Pegon -- Phase 06: Pegon Glyph Confusion Matrix
=============================================================================
Tujuan:
    Membangun confusion matrix antar karakter Pegon berdasarkan
    kemiripan visual glyph. Output digunakan untuk:
    1. Data augmentasi sintetis (Phase 07) — inject error realistis
    2. Knowledge Graph edges (Phase 08) — grapheme_confused_as

Metode (Guan & Greene, ACL 2024):
    Kombinasi tiga metode:
    - ORB  : keypoint matching
    - SIFT : gradient-based matching
    - Pixel IoU : shape overlap

    S_combined = 0.4*S_orb + 0.4*S_sift + 0.2*S_pixel
    P(i→j) = S_norm(i,j) / Σ_k S_norm(i,k)

Input : fonts/ (Amiri, NotoNaskh, Scheherazade)
Output: data/processed/confusion_matrix/
        ├── pegon_confusion_matrix.pkl
        ├── confusion_probs.pkl
        ├── pegon_confusion_matrix.csv
        ├── top_confusable_pairs.json
        └── confusion_matrix_heatmap.png

Run: python src/phase06_glyph_confusion.py
=============================================================================
"""

import os
import sys
import cv2
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR, PROCESSED_DIR, LOGS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(PROCESSED_DIR, "confusion_matrix")

FONT_CANDIDATES = [
    os.path.join(BASE_DIR, "fonts", "Amiri-Regular.ttf"),
    os.path.join(BASE_DIR, "fonts", "NotoNaskhArabic-Regular.ttf"),
    os.path.join(BASE_DIR, "fonts", "Scheherazade-Regular.ttf"),
    os.path.join(BASE_DIR, "fonts", "NotoKufiArabic-Regular.ttf"),
    os.path.join(BASE_DIR, "fonts", "Amiri-Regular.otf"),
]

FONT_SIZES      = [64, 96, 128]
CANVAS_SIZE     = 192
ORB_FEATURES    = 500
SIFT_RATIO      = 0.75
W_ORB           = 0.40
W_SIFT          = 0.40
W_PIXEL         = 0.20
CONF_THRESHOLD  = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# INVENTORI KARAKTER PEGON
# ─────────────────────────────────────────────────────────────────────────────

PEGON_CHARS = {
    # Konsonan Arab standar
    'alif'         : '\u0627',  # ا
    'ba'           : '\u0628',  # ب
    'ta'           : '\u062a',  # ت
    'tsa'          : '\u062b',  # ث
    'jim'          : '\u062c',  # ج
    'ha_kecil'     : '\u062d',  # ح
    'kha'          : '\u062e',  # خ
    'dal'          : '\u062f',  # د
    'dzal'         : '\u0630',  # ذ
    'ra'           : '\u0631',  # ر
    'zain'         : '\u0632',  # ز
    'sin'          : '\u0633',  # س
    'syin'         : '\u0634',  # ش
    'shad'         : '\u0635',  # ص
    'dhad'         : '\u0636',  # ض
    'tha'          : '\u0637',  # ط
    'zha'          : '\u0638',  # ظ
    'ain'          : '\u0639',  # ع
    'ghain'        : '\u063a',  # غ
    'fa'           : '\u0641',  # ف
    'qaf'          : '\u0642',  # ق
    'kaf'          : '\u0643',  # ك
    'lam'          : '\u0644',  # ل
    'mim'          : '\u0645',  # م
    'nun'          : '\u0646',  # ن
    'ha_besar'     : '\u0647',  # ه
    'waw'          : '\u0648',  # و
    'ya'           : '\u064a',  # ي
    'ta_marbuta'   : '\u0629',  # ة
    'alif_maqsura' : '\u0649',  # ى
    'hamza'        : '\u0621',  # ء
    'alif_hamza'   : '\u0623',  # أ
    'alif_madda'   : '\u0622',  # آ
    # Karakter khusus Pegon
    'pa'           : '\u06a4',  # ڤ
    'nga'          : '\u06a0',  # ڠ
    'ga'           : '\u06ae',  # ڮ
    'ca'           : '\u0684',  # ڄ
    'nya'          : '\u06bd',  # ڽ
    'nya2'         : '\u06d1',  # ۑ
}

CHAR_TO_NAME = {v: k for k, v in PEGON_CHARS.items()}
NAME_TO_CHAR = PEGON_CHARS

# ─────────────────────────────────────────────────────────────────────────────
# FONT LOADER
# ─────────────────────────────────────────────────────────────────────────────

def find_available_fonts(candidates):
    found = []
    for path in candidates:
        if os.path.isfile(path):
            found.append(path)
            print(f"  ✅ {Path(path).name}")
        else:
            print(f"  ❌ {Path(path).name} (not found)")
    return found


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTER RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def render_arabic_char(char, font, canvas=CANVAS_SIZE):
    """Render satu karakter Arab ke grayscale numpy array."""
    img  = Image.new('L', (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)

    display_char = char
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped     = arabic_reshaper.reshape(char)
        display_char = get_display(reshaped)
    except ImportError:
        pass

    try:
        bbox = draw.textbbox((0, 0), display_char, font=font)
    except AttributeError:
        w, h = draw.textsize(display_char, font=font)
        bbox = (0, 0, w, h)

    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]
    x = max(0, (canvas - char_w) // 2 - bbox[0])
    y = max(0, (canvas - char_h) // 2 - bbox[1])

    draw.text((x, y), display_char, font=font, fill=0)
    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY METHODS
# ─────────────────────────────────────────────────────────────────────────────

def pixel_iou(img1, img2, threshold=128):
    b1 = (img1 < threshold).astype(np.float32)
    b2 = (img2 < threshold).astype(np.float32)
    intersection = (b1 * b2).sum()
    union = np.maximum(b1, b2).sum()
    return float(intersection / union) if union > 1e-6 else 0.0


def orb_similarity(img1, img2):
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0.0
    if len(des1) < 2 or len(des2) < 2:
        return 0.0
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    return len(matches) / max(len(kp1), len(kp2))


def sift_similarity(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0.0
    if len(des1) < 2 or len(des2) < 2:
        return 0.0
    bf      = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < SIFT_RATIO * n.distance]
    return len(good) / max(len(kp1), len(kp2))


def combined_similarity(img1, img2):
    s_orb   = orb_similarity(img1, img2)
    s_sift  = sift_similarity(img1, img2)
    s_pixel = pixel_iou(img1, img2)
    return W_ORB * s_orb + W_SIFT * s_sift + W_PIXEL * s_pixel


# ─────────────────────────────────────────────────────────────────────────────
# BUILD CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix(chars, font_paths, font_sizes=FONT_SIZES):
    char_list  = list(chars.values())
    char_names = list(chars.keys())
    n          = len(char_list)

    print(f"\n  Building {n}x{n} matrix")
    print(f"  Fonts : {len(font_paths)} | Sizes: {font_sizes}")
    print(f"  Total combinations: {len(font_paths) * len(font_sizes)}")

    # scores[i][j] = list of similarity values dari semua font+size
    scores = [[[] for _ in range(n)] for _ in range(n)]
    total_combos = len(font_paths) * len(font_sizes)
    combo_idx    = 0

    for font_path in font_paths:
        font_name = Path(font_path).stem
        for size in font_sizes:
            combo_idx += 1
            print(f"  [{combo_idx}/{total_combos}] {font_name} size={size}px...",
                  end=" ", flush=True)

            font    = ImageFont.truetype(font_path, size)
            renders = [render_arabic_char(ch, font) for ch in char_list]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        scores[i][j].append(1.0)
                    else:
                        scores[i][j].append(
                            combined_similarity(renders[i], renders[j])
                        )
            print(f"done")

    # Rata-ratakan semua font+size
    avg_matrix = {}
    for i, name_i in enumerate(char_names):
        avg_matrix[name_i] = {}
        for j, name_j in enumerate(char_names):
            avg_matrix[name_i][name_j] = round(
                float(np.mean(scores[i][j])), 6
            )

    return avg_matrix, char_names, char_list


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION PROBABILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion_probs(raw_scores, threshold=CONF_THRESHOLD):
    probs = {}
    for char_i, row in raw_scores.items():
        others       = {c: s for c, s in row.items() if c != char_i}
        above_thresh = {c: max(0.0, s - threshold) for c, s in others.items()}
        total        = sum(above_thresh.values())
        if total < 1e-10:
            n = len(others)
            probs[char_i] = {c: 1.0/n for c in others}
        else:
            probs[char_i] = {c: v/total for c, v in above_thresh.items()}
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(raw_scores, char_names, char_list, output_path):
    n      = len(char_names)
    matrix = np.zeros((n, n))
    for i, ni in enumerate(char_names):
        for j, nj in enumerate(char_names):
            matrix[i][j] = raw_scores[ni].get(nj, 0.0)
    np.fill_diagonal(matrix, 0)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(matrix, cmap='hot_r', aspect='auto', vmin=0, vmax=0.8)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Visual Similarity Score')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(char_list, fontsize=8, rotation=90)
    ax.set_yticklabels(char_list, fontsize=8)
    ax.set_title('Pegon Glyph Confusion Matrix\n(diagonal=0 for visibility)',
                 fontsize=13, pad=15)
    ax.set_xlabel('Confused As', fontsize=11)
    ax.set_ylabel('Original Character', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Heatmap: {output_path}")


def find_top_pairs(raw_scores, char_names, top_k=30):
    pairs = []
    names = list(raw_scores.keys())
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i >= j:
                continue
            sim = raw_scores[ni].get(nj, 0.0)
            pairs.append({
                'char_i'    : PEGON_CHARS.get(ni, ni),
                'char_j'    : PEGON_CHARS.get(nj, nj),
                'name_i'    : ni,
                'name_j'    : nj,
                'similarity': round(sim, 4),
            })
    pairs.sort(key=lambda x: -x['similarity'])
    return pairs[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 60)
    print("CASTLE-Pegon -- Phase 06: Glyph Confusion Matrix")
    print("=" * 60)
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Chars  : {len(PEGON_CHARS)}")

    # Install dependencies jika belum ada
    try:
        import arabic_reshaper
    except ImportError:
        print("\nInstalling arabic_reshaper...")
        os.system("pip install arabic-reshaper python-bidi -q")

    # Find fonts
    print("\nSearching fonts...")
    available_fonts = find_available_fonts(FONT_CANDIDATES)
    if not available_fonts:
        print("❌ Tidak ada font ditemukan!")
        print(f"   Pastikan font ada di: {os.path.join(BASE_DIR, 'fonts')}/")
        return

    # Pakai max 3 font untuk efisiensi
    fonts_to_use = available_fonts[:3]
    print(f"\nUsing {len(fonts_to_use)} font(s) x {len(FONT_SIZES)} sizes")

    # Build confusion matrix
    print("\nBuilding confusion matrix (estimasi 3-5 menit)...")
    raw_scores, char_names, char_list = build_confusion_matrix(
        PEGON_CHARS, fonts_to_use, FONT_SIZES
    )

    # Confusion probabilities
    print("\nComputing confusion probabilities...")
    probs = compute_confusion_probs(raw_scores)

    # Top confusable pairs
    top_pairs = find_top_pairs(raw_scores, char_names, top_k=30)

    print(f"\nTop 20 confusable pairs:")
    print(f"  {'Rank':<5} {'Pair':<10} {'Names':<35} {'Sim':>6}")
    print(f"  {'─'*60}")
    for rank, p in enumerate(top_pairs[:20], 1):
        pair  = f"{p['char_i']} ↔ {p['char_j']}"
        names = f"{p['name_i']} / {p['name_j']}"
        print(f"  {rank:<5} {pair:<10} {names:<35} {p['similarity']:>6.4f}")

    # Save outputs
    print(f"\nSaving outputs to {OUTPUT_DIR}...")

    # 1. Raw matrix pickle
    matrix_path = os.path.join(OUTPUT_DIR, "pegon_confusion_matrix.pkl")
    with open(matrix_path, 'wb') as f:
        pickle.dump({
            'raw_scores'  : raw_scores,
            'char_names'  : char_names,
            'char_list'   : char_list,
            'char_to_name': CHAR_TO_NAME,
            'name_to_char': NAME_TO_CHAR,
            'fonts_used'  : [Path(f).name for f in fonts_to_use],
            'font_sizes'  : FONT_SIZES,
        }, f)
    print(f"  ✅ pegon_confusion_matrix.pkl")

    # 2. Confusion probs pickle
    probs_path = os.path.join(OUTPUT_DIR, "confusion_probs.pkl")
    with open(probs_path, 'wb') as f:
        pickle.dump({
            'probs'       : probs,
            'char_to_name': CHAR_TO_NAME,
            'name_to_char': NAME_TO_CHAR,
            'threshold'   : CONF_THRESHOLD,
        }, f)
    print(f"  ✅ confusion_probs.pkl")

    # 3. CSV
    csv_path = os.path.join(OUTPUT_DIR, "pegon_confusion_matrix.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['char_name', 'char'] + char_names)
        for ni in char_names:
            row = [ni, PEGON_CHARS.get(ni, ni)]
            for nj in char_names:
                row.append(f"{raw_scores[ni][nj]:.4f}")
            writer.writerow(row)
    print(f"  ✅ pegon_confusion_matrix.csv")

    # 4. Top pairs JSON
    top_path = os.path.join(OUTPUT_DIR, "top_confusable_pairs.json")
    with open(top_path, 'w', encoding='utf-8') as f:
        json.dump(top_pairs, f, ensure_ascii=False, indent=2)
    print(f"  ✅ top_confusable_pairs.json")

    # 5. Heatmap
    heatmap_path = os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png")
    try:
        plot_heatmap(raw_scores, char_names, char_list, heatmap_path)
    except Exception as e:
        print(f"  ⚠️  Heatmap failed: {e}")

    # Summary stats
    all_scores = [
        raw_scores[ni][nj]
        for ni in char_names
        for nj in char_names
        if ni != nj
    ]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Matrix   : {len(char_names)} x {len(char_names)}")
    print(f"  Sim range: {min(all_scores):.4f} – {max(all_scores):.4f}")
    print(f"  Sim mean : {np.mean(all_scores):.4f}")
    print(f"  Pairs>0.5: {sum(1 for s in all_scores if s > 0.5)}")
    print(f"  Pairs>0.3: {sum(1 for s in all_scores if s > 0.3)}")
    print(f"\n  confusion_probs.pkl → dipakai Phase 07 (augmentation)")
    print(f"  pegon_confusion_matrix.pkl → dipakai Phase 08 (KG edges)")
    print(f"\n  NEXT: python src/phase07_augment.py")
    print("=" * 60)


if __name__ == "__main__":
    main()