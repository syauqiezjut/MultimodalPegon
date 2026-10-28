"""
=============================================================================
PHASE 3 — Pegon Glyph Confusion Matrix
=============================================================================
CASTLE-Pegon Paper 4 | Syauqie Muhammad Marier | ZJUT 2026

Tujuan:
    Membangun matrix kebingungan (confusion matrix) antar karakter Pegon
    berdasarkan kemiripan visual glyph. Matrix ini digunakan untuk:
    1. Data augmentasi sintetis (Phase 3.2) — injeksi error realistis
    2. Knowledge Graph edges (Phase 4) — grapheme_confused_as
    3. Analisis error di paper — "mengapa OCR salah pada karakter ini"

Metode (dari Guan & Greene, ACL 2024):
    Kombinasi tiga metode feature matching:
    - ORB  (Oriented FAST and Rotated BRIEF)  — keypoint matching
    - SIFT (Scale-Invariant Feature Transform) — gradient matching
    - Pixel IoU                               — shape overlap

    S_combined(i,j) = 0.4*S_orb + 0.4*S_sift + 0.2*S_pixel

    Confusion probability:
    P(i→j) = S_norm(i,j) / Σ_k S_norm(i,k)

OUTPUT:
    results/confusion_matrix/
    ├── pegon_confusion_matrix.pkl      ← matrix lengkap (pakai di Phase 4)
    ├── pegon_confusion_matrix.csv      ← untuk inspeksi manual
    ├── confusion_probs.pkl             ← probability matrix (pakai di augmentasi)
    ├── top_confusable_pairs.json       ← 20 pasang paling mirip
    └── confusion_matrix_heatmap.png    ← visualisasi

INSTALL:
    pip install opencv-python pillow numpy scipy matplotlib arabic-reshaper python-bidi
    # Font: Amiri, Scheherazade New, Noto Naskh Arabic
    # Windows: copy ke C:/Windows/Fonts/
    # Linux:   fc-cache -fv setelah copy ke ~/.fonts/
=============================================================================
"""

import os
import cv2
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "./results/confusion_matrix"

# ── Font paths — SESUAIKAN DENGAN SISTEM ANDA ──
# Script akan mencoba setiap font dalam urutan ini, pakai yang pertama ditemukan
FONT_CANDIDATES = [
    # Linux paths
    "./fonts/Amiri-Regular.ttf",
    "./fonts/NotoKufiArabic-Regular.ttf",
    "./fonts/NotoNaskhArabic-Regular.ttf",
    "./fonts/Scheherazade-Regular.ttf"
]

# Font sizes untuk rendering (multiple sizes = lebih robust)
FONT_SIZES = [64, 96, 128]

# Canvas size untuk setiap karakter (square)
CANVAS_SIZE = 192

# ORB/SIFT parameters
ORB_FEATURES  = 500
SIFT_RATIO    = 0.75   # Lowe's ratio test threshold

# Bobot kombinasi similarity
W_ORB   = 0.40
W_SIFT  = 0.40
W_PIXEL = 0.20

# Threshold: pasang dengan similarity < ini dianggap tidak confusable
CONFUSION_THRESHOLD = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# INVENTORI KARAKTER PEGON
# ─────────────────────────────────────────────────────────────────────────────

# Karakter yang akan dimasukkan ke confusion matrix
# Dikategorikan untuk analisis breakdown di paper
PEGON_CHARS = {

    # ── Konsonan Arab standar (shared dengan Al-Quran / teks Arab) ──
    'alif'           : 'ا',
    'ba'             : 'ب',
    'ta'             : 'ت',
    'tsa'            : 'ث',
    'jim'            : 'ج',
    'ha_kecil'       : 'ح',
    'kha'            : 'خ',
    'dal'            : 'د',
    'dzal'           : 'ذ',
    'ra'             : 'ر',
    'zain'           : 'ز',
    'sin'            : 'س',
    'syin'           : 'ش',
    'shad'           : 'ص',
    'dhad'           : 'ض',
    'tha'            : 'ط',
    'zha'            : 'ظ',
    'ain'            : 'ع',
    'ghain'          : 'غ',
    'fa'             : 'ف',
    'qaf'            : 'ق',
    'kaf'            : 'ك',
    'lam'            : 'ل',
    'mim'            : 'م',
    'nun'            : 'ن',
    'ha_besar'       : 'ه',
    'waw'            : 'و',
    'ya'             : 'ي',
    'ta_marbuta'     : 'ة',
    'alif_maqsura'   : 'ى',
    'hamza'          : 'ء',
    'alif_hamza'     : 'أ',
    'alif_madda'     : 'آ',
    'lam_alif'       : 'لا',

    # ── Karakter khusus Pegon (fonem Jawa/Sunda) ──
    'pa'             : 'ڤ',   # U+06A4 — pa, fonem /p/
    'nga'            : 'ڠ',   # U+06A0 — nga, fonem /ŋ/
    'ga'             : 'ڮ',   # U+06AE — ga (versi Pegon dari ك)
    'ca'             : 'ڄ',   # U+0684 — ca, fonem /tʃ/
    'nya'            : 'ڽ',   # U+06BD — nya, fonem /ɲ/
    'nya2'           : 'ۑ',   # U+06D1 — nya alternatif
}

# Grup karakter yang secara visual sangat mirip
# (untuk analisis di paper Section 3 dan error analysis)
VISUAL_GROUPS = {
    'dot_family'        : ['ب','ت','ث','ن','ي'],      # sama base, beda titik
    'hook_family'       : ['ح','ج','خ'],               # base sama persis
    'dal_family'        : ['د','ذ'],                   # 1 titik perbedaan
    'ra_family'         : ['ر','ز'],                   # 1 titik perbedaan
    'sin_family'        : ['س','ش'],                   # 3 titik di atas
    'shad_family'       : ['ص','ض'],                   # 1 titik perbedaan
    'tha_family'        : ['ط','ظ'],                   # 1 titik perbedaan
    'ain_family'        : ['ع','غ'],                   # 1 titik perbedaan
    'fa_family'         : ['ف','ق'],                   # 1 vs 2 titik
    'pegon_pa_group'    : ['ڤ','ف','ق'],               # Pegon pa vs fa/qaf
    'pegon_nga_group'   : ['ڠ','ع','غ'],               # Pegon nga vs ain/ghain
    'pegon_ga_group'    : ['ڮ','ك'],                   # Pegon ga vs kaf
    'pegon_ca_group'    : ['ڄ','ج','خ'],               # Pegon ca vs jim/kha
    'round_family'      : ['و','ر','ز'],               # bentuk bulat
}


# ─────────────────────────────────────────────────────────────────────────────
# FONT LOADER
# ─────────────────────────────────────────────────────────────────────────────

def find_available_fonts(candidates: list) -> list:
    """Temukan semua font yang tersedia dari daftar kandidat."""
    found = []
    for path in candidates:
        if os.path.isfile(path):
            found.append(path)
    return found


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load font dengan fallback."""
    return ImageFont.truetype(font_path, size)


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTER RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def render_arabic_char(char: str, font: ImageFont.FreeTypeFont,
                       canvas: int = CANVAS_SIZE) -> np.ndarray:
    """
    Render satu karakter Arab ke grayscale image.

    Karakter Arab memerlukan:
    1. Konteks untuk contextual shaping (isolated form)
    2. Right-to-left rendering

    Strategi: render dalam konteks isolated menggunakan zero-width joiner
    atau render langsung dan crop ke bounding box.

    Returns: numpy array (canvas x canvas), uint8, 0=black 255=white
    """
    img  = Image.new('L', (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)

    # Coba render dengan arabic_reshaper jika tersedia
    display_char = char
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(char)
        display_char = get_display(reshaped)
    except ImportError:
        pass  # render as-is

    # Hitung bounding box untuk centering
    try:
        bbox = draw.textbbox((0, 0), display_char, font=font)
    except AttributeError:
        # PIL versi lama
        w, h = draw.textsize(display_char, font=font)
        bbox = (0, 0, w, h)

    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    # Center dalam canvas
    x = max(0, (canvas - char_w) // 2 - bbox[0])
    y = max(0, (canvas - char_h) // 2 - bbox[1])

    draw.text((x, y), display_char, font=font, fill=0)
    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY METHODS
# ─────────────────────────────────────────────────────────────────────────────

def pixel_iou(img1: np.ndarray, img2: np.ndarray,
              threshold: int = 128) -> float:
    """
    Pixel-level Intersection over Union.
    Binarize dengan threshold, hitung overlap.
    """
    b1 = (img1 < threshold).astype(np.float32)
    b2 = (img2 < threshold).astype(np.float32)
    intersection = (b1 * b2).sum()
    union = np.maximum(b1, b2).sum()
    return float(intersection / union) if union > 1e-6 else 0.0


def orb_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    ORB keypoint matching similarity.
    Returns: ratio of good matches to max possible matches.
    """
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

    max_kp = max(len(kp1), len(kp2))
    return len(matches) / max_kp


def sift_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    SIFT keypoint matching similarity with Lowe's ratio test.
    """
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

    max_kp = max(len(kp1), len(kp2))
    return len(good) / max_kp


def combined_similarity(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Hitung similarity gabungan dari ORB, SIFT, dan Pixel IoU.
    Returns dict dengan semua skor individual dan combined.
    """
    s_orb   = orb_similarity(img1, img2)
    s_sift  = sift_similarity(img1, img2)
    s_pixel = pixel_iou(img1, img2)
    s_comb  = W_ORB * s_orb + W_SIFT * s_sift + W_PIXEL * s_pixel

    return {
        'orb'      : round(s_orb,   6),
        'sift'     : round(s_sift,  6),
        'pixel_iou': round(s_pixel, 6),
        'combined' : round(s_comb,  6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix(chars: dict, font_paths: list,
                           font_sizes: list = FONT_SIZES) -> dict:
    """
    Bangun confusion matrix lengkap untuk semua pasang karakter.

    Returns:
        raw_scores[char_i][char_j] = dict(orb, sift, pixel_iou, combined)
        Dirata-ratakan dari semua kombinasi font x size.
    """
    char_list  = list(chars.values())
    char_names = list(chars.keys())
    n          = len(char_list)

    print(f"\nBuilding {n}x{n} confusion matrix")
    print(f"Fonts  : {len(font_paths)} fonts")
    print(f"Sizes  : {font_sizes}")
    print(f"Total pairs: {n*n}")

    # Inisialisasi akumulator
    # scores[i][j] = list of combined similarity values
    scores = [[[] for _ in range(n)] for _ in range(n)]

    total_combos = len(font_paths) * len(font_sizes)
    combo_idx    = 0

    for font_path in font_paths:
        font_name = Path(font_path).stem
        for size in font_sizes:
            combo_idx += 1
            print(f"\n  [{combo_idx}/{total_combos}] Font={font_name}  Size={size}px")

            # Render semua karakter dengan font+size ini
            font    = load_font(font_path, size)
            renders = []
            for ch in char_list:
                img = render_arabic_char(ch, font)
                renders.append(img)
            print(f"    Rendered {len(renders)} characters")

            # Hitung similarity semua pasang
            for i in range(n):
                for j in range(n):
                    if i == j:
                        scores[i][j].append(1.0)  # same char = similarity 1
                        continue
                    sim = combined_similarity(renders[i], renders[j])
                    scores[i][j].append(sim['combined'])

            # Progress
            pairs_done = n * n
            print(f"    {pairs_done} pairs computed")

    # Rata-ratakan dari semua font+size
    avg_matrix = {}
    for i, name_i in enumerate(char_names):
        avg_matrix[name_i] = {}
        for j, name_j in enumerate(char_names):
            avg = float(np.mean(scores[i][j]))
            avg_matrix[name_i][name_j] = round(avg, 6)

    return avg_matrix, char_names, char_list


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion_probs(raw_scores: dict, threshold: float = CONFUSION_THRESHOLD) -> dict:
    """
    Konversi raw similarity scores ke confusion probabilities.

    Formula (Guan & Greene 2024):
        S_norm(i,j) = max(0, S(i,j) - threshold)
        P(i→j) = S_norm(i,j) / Σ_k S_norm(i,k)   (k ≠ i)

    Returns: probs[char_i][char_j] = probability of i being misread as j
    """
    probs = {}

    for char_i, row in raw_scores.items():
        # Ambil semua skor kecuali diri sendiri
        others = {c: s for c, s in row.items() if c != char_i}

        # Threshold dan normalize
        above_thresh = {c: max(0.0, s - threshold) for c, s in others.items()}
        total        = sum(above_thresh.values())

        if total < 1e-10:
            # Tidak ada karakter yang mirip — distribusi uniform
            n = len(others)
            probs[char_i] = {c: 1.0/n for c in others}
        else:
            probs[char_i] = {c: v/total for c, v in above_thresh.items()}

    return probs


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def augment_with_confusion(text: str, probs: dict,
                           char_to_name: dict,
                           name_to_char: dict,
                           error_rate: float = 0.15) -> str:
    """
    Injeksi error realistis ke dalam teks Pegon menggunakan confusion matrix.
    Setiap karakter punya probabilitas error_rate untuk diganti.

    Digunakan di Phase 3.2 untuk membuat synthetic training data.
    """
    import random
    result = []
    for char in text:
        name = char_to_name.get(char)
        if name and name in probs and random.random() < error_rate:
            # Sample confusion target berdasarkan probability
            targets = list(probs[name].keys())
            weights = list(probs[name].values())
            if targets and sum(weights) > 0:
                chosen_name = random.choices(targets, weights=weights, k=1)[0]
                result.append(name_to_char.get(chosen_name, char))
            else:
                result.append(char)
        else:
            result.append(char)
    return ''.join(result)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_heatmap(raw_scores: dict, char_names: list,
                           char_list: list, output_path: str):
    """
    Plot heatmap confusion matrix — karakter dengan label Unicode.
    """
    n     = len(char_names)
    matrix = np.zeros((n, n))

    name_to_idx = {name: i for i, name in enumerate(char_names)}
    for i, name_i in enumerate(char_names):
        for j, name_j in enumerate(char_names):
            matrix[i][j] = raw_scores[name_i].get(name_j, 0.0)

    # Set diagonal to 0 untuk visualisasi yang lebih baik
    np.fill_diagonal(matrix, 0)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(matrix, cmap='hot_r', aspect='auto', vmin=0, vmax=0.8)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Visual Similarity Score')

    # Labels menggunakan karakter Unicode langsung
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(char_list, fontsize=9, rotation=90)
    ax.set_yticklabels(char_list, fontsize=9)

    ax.set_title('Pegon Glyph Confusion Matrix\n(diagonal set to 0 for visibility)',
                 fontsize=13, pad=15)
    ax.set_xlabel('Confused As →', fontsize=11)
    ax.set_ylabel('Original Character', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {output_path}")


def find_top_confusable_pairs(raw_scores: dict, char_names: list,
                              top_k: int = 20) -> list:
    """Temukan top-K pasang karakter paling mudah membingungkan."""
    pairs = []
    names = list(raw_scores.keys())
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i >= j:
                continue
            sim = raw_scores[name_i].get(name_j, 0.0)
            pairs.append({
                'char_i'     : PEGON_CHARS.get(name_i, name_i),
                'char_j'     : PEGON_CHARS.get(name_j, name_j),
                'name_i'     : name_i,
                'name_j'     : name_j,
                'similarity' : round(sim, 4),
            })

    pairs.sort(key=lambda x: -x['similarity'])
    return pairs[:top_k]


def print_top_pairs(pairs: list):
    """Print top confusable pairs dengan formatting rapi."""
    print(f"\n{'─'*55}")
    print(f"TOP CONFUSABLE PAIRS")
    print(f"{'─'*55}")
    print(f"  {'Rank':<5} {'Pair':<8} {'Names':<30} {'Similarity':>10}")
    for rank, p in enumerate(pairs, 1):
        pair_str = f"{p['char_i']} ↔ {p['char_j']}"
        names    = f"{p['name_i']} / {p['name_j']}"
        print(f"  {rank:<5} {pair_str:<8} {names:<30} {p['similarity']:>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("CASTLE-Pegon — Phase 3: Glyph Confusion Matrix")
    print("=" * 60)

    # ── Cari font yang tersedia ──
    print("\nSearching for Arabic fonts...")
    available_fonts = find_available_fonts(FONT_CANDIDATES)

    if not available_fonts:
        print("❌ Tidak ada font Arab yang ditemukan!")
        print("\nInstall salah satu dari:")
        print("  # Ubuntu/Debian:")
        print("  sudo apt install fonts-amiri fonts-sil-scheherazade fonts-noto-core")
        print("\n  # Manual download:")
        print("  https://fonts.google.com/specimen/Amiri")
        print("  https://software.sil.org/scheherazade/")
        print("  Copy .ttf ke ~/.fonts/ lalu jalankan: fc-cache -fv")
        return

    print(f"Found {len(available_fonts)} font(s):")
    for f in available_fonts:
        print(f"  ✅ {Path(f).name}")

    # Batasi ke 3 font terbaik (lebih banyak = lebih lambat)
    fonts_to_use = available_fonts[:3]

    # ── Char maps ──
    char_to_name = {v: k for k, v in PEGON_CHARS.items()}
    name_to_char = PEGON_CHARS

    print(f"\nCharacter inventory: {len(PEGON_CHARS)} characters")

    # ── Build confusion matrix ──
    print("\nBuilding confusion matrix...")
    print("(Ini mungkin butuh 2-5 menit tergantung GPU/CPU)")

    raw_scores, char_names, char_list = build_confusion_matrix(
        PEGON_CHARS,
        fonts_to_use,
        FONT_SIZES,
    )

    # ── Confusion probabilities ──
    print("\nComputing confusion probabilities...")
    probs = compute_confusion_probs(raw_scores)

    # ── Top confusable pairs ──
    top_pairs = find_top_confusable_pairs(raw_scores, char_names, top_k=30)
    print_top_pairs(top_pairs[:20])

    # ── Validate augmentation ──
    print(f"\n{'─'*55}")
    print("AUGMENTATION TEST")
    print(f"{'─'*55}")
    test_text = "بسم الله الرحمن الرحيم"
    augmented = augment_with_confusion(
        test_text, probs, char_to_name, name_to_char, error_rate=0.3
    )
    print(f"  Original  : {test_text}")
    print(f"  Augmented : {augmented}")
    print(f"  (error_rate=0.3 — 30% chars replaced)")

    # ── Save outputs ──
    print(f"\n{'─'*55}")
    print("SAVING OUTPUTS")
    print(f"{'─'*55}")

    # 1. Raw scores pickle
    matrix_path = os.path.join(OUTPUT_DIR, "pegon_confusion_matrix.pkl")
    with open(matrix_path, 'wb') as f:
        pickle.dump({
            'raw_scores'  : raw_scores,
            'char_names'  : char_names,
            'char_list'   : char_list,
            'char_to_name': char_to_name,
            'name_to_char': name_to_char,
            'fonts_used'  : [Path(f).name for f in fonts_to_use],
            'font_sizes'  : FONT_SIZES,
        }, f)
    print(f"  ✅ {matrix_path}")

    # 2. Confusion probabilities pickle
    probs_path = os.path.join(OUTPUT_DIR, "confusion_probs.pkl")
    with open(probs_path, 'wb') as f:
        pickle.dump({
            'probs'       : probs,
            'char_to_name': char_to_name,
            'name_to_char': name_to_char,
            'threshold'   : CONFUSION_THRESHOLD,
        }, f)
    print(f"  ✅ {probs_path}")

    # 3. CSV untuk inspeksi manual
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "pegon_confusion_matrix.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['char_name', 'char'] + char_names
        writer.writerow(header)
        for name_i in char_names:
            char_i = PEGON_CHARS.get(name_i, name_i)
            row    = [name_i, char_i]
            for name_j in char_names:
                row.append(f"{raw_scores[name_i][name_j]:.4f}")
            writer.writerow(row)
    print(f"  ✅ {csv_path}")

    # 4. Top pairs JSON
    top_path = os.path.join(OUTPUT_DIR, "top_confusable_pairs.json")
    with open(top_path, 'w', encoding='utf-8') as f:
        json.dump(top_pairs, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {top_path}")

    # 5. Heatmap
    heatmap_path = os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png")
    try:
        plot_confusion_heatmap(raw_scores, char_names, char_list, heatmap_path)
        print(f"  ✅ {heatmap_path}")
    except Exception as e:
        print(f"  ⚠️  Heatmap failed: {e}")

    # ── Summary stats ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_scores = [
        raw_scores[n_i][n_j]
        for n_i in char_names for n_j in char_names
        if n_i != n_j
    ]
    print(f"  Matrix size   : {len(char_names)} x {len(char_names)}")
    print(f"  Similarity range: {min(all_scores):.4f} – {max(all_scores):.4f}")
    print(f"  Mean similarity : {np.mean(all_scores):.4f}")
    print(f"  Pairs > 0.5     : {sum(1 for s in all_scores if s > 0.5)}")
    print(f"  Pairs > 0.3     : {sum(1 for s in all_scores if s > 0.3)}")
    print(f"  Outputs saved to: {os.path.abspath(OUTPUT_DIR)}")

    print(f"\n{'='*60}")
    print("FILES GENERATED:")
    print("  pegon_confusion_matrix.pkl  ← load di Phase 4 (Knowledge Graph)")
    print("  confusion_probs.pkl         ← load di Phase 3.2 (augmentation)")
    print("  pegon_confusion_matrix.csv  ← buka di Excel untuk inspeksi")
    print("  top_confusable_pairs.json   ← pakai di paper Table 2")
    print("  confusion_matrix_heatmap.png← pakai di paper Figure 3")
    print(f"{'='*60}")
    print()
    print("NEXT STEP: Phase 3.2 — Synthetic Data Augmentation")
    print("  Gunakan confusion_probs.pkl untuk inject error ke ground_truth")
    print("  Target: 4x augmentasi dari ~700 train lines → ~2800 training pairs")


if __name__ == "__main__":
    main()