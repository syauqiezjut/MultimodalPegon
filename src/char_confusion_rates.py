"""
=============================================================================
Character-Level Confusion Rates Before/After Correction
=============================================================================
Computes how often each confusable pair appears in errors, comparing:
  - GPT-4o output (before correction)
  - v2 output (after multimodal correction)

Shows that v2 specifically reduces Pegon-specific confusion pairs.

Run on AutoDL:
    python char_confusion_rates.py

Output: char_confusion_analysis.txt + confusion_before_after.pdf/png
=============================================================================
"""

import json
import os
import unicodedata
import numpy as np
from collections import Counter, defaultdict

# ── Paths ──
BASE = "/root/autodl-tmp/castle-pegon"
GPT4O_PATH = f"{BASE}/results/ocr/tier3_gpt_gpt_4o_predictions.json"
V2_PATH = f"{BASE}/results/step1v2_finetuned_vit/predictions_test.json"
OUTPUT_DIR = f"{BASE}/results/char_confusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Pegon characters & known confusable pairs ──
PEGON_PAIRS = {
    ('ف', 'ڤ'): 'fa→pa',     # Most common
    ('ج', 'ڄ'): 'jim→ca',
    ('غ', 'ڠ'): 'ghain→nga',
    ('ك', 'ڮ'): 'kaf→ga',
    ('ن', 'ڽ'): 'nun→nya',
}

ARABIC_DOT_PAIRS = {
    ('ط', 'ظ'): 'tha→zha',
    ('ص', 'ض'): 'shad→dhad',
    ('ع', 'غ'): 'ain→ghain',
    ('ح', 'خ'): 'ha→kha',
    ('د', 'ذ'): 'dal→dzal',
    ('ر', 'ز'): 'ra→zain',
    ('ب', 'ت'): 'ba→ta',
    ('ت', 'ث'): 'ta→tsa',
    ('س', 'ش'): 'sin→syin',
    ('ج', 'ح'): 'jim→ha',
}

ALL_TRACKED_PAIRS = {**PEGON_PAIRS, **ARABIC_DOT_PAIRS}


def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\u0640", "")
    return " ".join(text.split()).strip()


def align_and_count_substitutions(pred_text, ref_text):
    """
    Align pred and ref using edit distance backtrack,
    then count character substitutions.
    Returns: list of (pred_char, ref_char) substitution pairs
    """
    pred = normalize_text(pred_text)
    ref = normalize_text(ref_text)
    
    m, n = len(pred), len(ref)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                    dp[i][j-1],      # insertion
                                    dp[i-1][j-1])    # substitution
    
    # Backtrack to find substitutions
    substitutions = []
    i, j = m, n
    while i > 0 and j > 0:
        if pred[i-1] == ref[j-1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            substitutions.append((pred[i-1], ref[j-1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            # Deletion (extra char in pred)
            i -= 1
        else:
            # Insertion (missing char in pred)
            j -= 1
    
    return substitutions


def count_pair_confusions(substitutions, tracked_pairs):
    """Count how often each tracked pair appears in substitutions."""
    counts = Counter()
    
    for pred_char, ref_char in substitutions:
        pair = (pred_char, ref_char)
        reverse_pair = (ref_char, pred_char)
        
        if pair in tracked_pairs:
            counts[pair] += 1
        elif reverse_pair in tracked_pairs:
            counts[reverse_pair] += 1
    
    return counts


def main():
    print("=" * 65)
    print("  Character-Level Confusion Rates Before/After Correction")
    print("=" * 65)
    
    # Load data
    with open(GPT4O_PATH, 'r', encoding='utf-8') as f:
        gpt4o_data = json.load(f)
    with open(V2_PATH, 'r', encoding='utf-8') as f:
        v2_data = json.load(f)
    
    gpt4o_dict = {r['record_id']: r for r in gpt4o_data}
    v2_dict = {r['record_id']: r for r in v2_data}
    common_ids = sorted(set(gpt4o_dict.keys()) & set(v2_dict.keys()))
    
    print(f"  Analyzing {len(common_ids)} test samples...")
    
    # Count substitutions
    gpt4o_total_counts = Counter()
    v2_total_counts = Counter()
    gpt4o_total_subs = 0
    v2_total_subs = 0
    
    for rid in common_ids:
        gpt = gpt4o_dict[rid]
        v2 = v2_dict[rid]
        ref = gpt.get('gt_normalized') or gpt.get('ground_truth', '')
        gpt_text = gpt.get('gpt_gpt_4o_norm') or gpt.get('gpt_gpt_4o', '')
        v2_text = v2.get('prediction', '')
        
        # GPT-4o substitutions
        gpt_subs = align_and_count_substitutions(gpt_text, ref)
        gpt4o_total_subs += len(gpt_subs)
        gpt4o_pair_counts = count_pair_confusions(gpt_subs, ALL_TRACKED_PAIRS)
        gpt4o_total_counts += gpt4o_pair_counts
        
        # v2 substitutions
        v2_subs = align_and_count_substitutions(v2_text, ref)
        v2_total_subs += len(v2_subs)
        v2_pair_counts = count_pair_confusions(v2_subs, ALL_TRACKED_PAIRS)
        v2_total_counts += v2_pair_counts
    
    # ── Report ──
    report_path = os.path.join(OUTPUT_DIR, "char_confusion_analysis.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CHARACTER-LEVEL CONFUSION RATES BEFORE/AFTER CORRECTION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total substitution errors — GPT-4o: {gpt4o_total_subs}  |  v2: {v2_total_subs}\n")
        f.write(f"Reduction: {gpt4o_total_subs - v2_total_subs} ({(1 - v2_total_subs/max(gpt4o_total_subs,1))*100:.1f}%)\n\n")
        
        f.write(f"{'Pair':<20} {'Label':<15} {'GPT-4o':>8} {'v2':>8} {'Δ':>8} {'Reduction':>10}\n")
        f.write("-" * 70 + "\n")
        
        # Sort: Pegon pairs first, then by GPT-4o count
        all_pairs = list(ALL_TRACKED_PAIRS.keys())
        pegon_keys = list(PEGON_PAIRS.keys())
        arabic_keys = list(ARABIC_DOT_PAIRS.keys())
        
        for section_name, section_pairs in [("PEGON-SPECIFIC PAIRS", pegon_keys), 
                                             ("ARABIC DOT PAIRS", arabic_keys)]:
            f.write(f"\n  {section_name}\n")
            for pair in section_pairs:
                label = ALL_TRACKED_PAIRS[pair]
                g_count = gpt4o_total_counts.get(pair, 0)
                v_count = v2_total_counts.get(pair, 0)
                delta = g_count - v_count
                reduction = (1 - v_count / max(g_count, 1)) * 100 if g_count > 0 else 0
                f.write(f"  {pair[0]}→{pair[1]:<15} {label:<15} {g_count:>8} {v_count:>8} {delta:>+8} {reduction:>9.1f}%\n")
        
        f.write("\n")
    
    print(f"\n  Report saved: {report_path}")
    
    # Print summary to console
    print(f"\n  ── PEGON-SPECIFIC CONFUSION PAIRS ──")
    print(f"  {'Pair':<12} {'GPT-4o':>8} {'v2':>8} {'Reduction':>10}")
    for pair in pegon_keys:
        label = PEGON_PAIRS[pair]
        g = gpt4o_total_counts.get(pair, 0)
        v = v2_total_counts.get(pair, 0)
        red = (1 - v/max(g,1))*100 if g > 0 else 0
        print(f"  {pair[0]}→{pair[1]:<8} {g:>8} {v:>8} {red:>9.1f}%")
    
    # ── Generate bar chart ──
    print(f"\n  Generating comparison chart...")
    generate_chart(pegon_keys, arabic_keys, gpt4o_total_counts, v2_total_counts)
    
    print(f"\n{'=' * 65}")


def generate_chart(pegon_keys, arabic_keys, gpt4o_counts, v2_counts):
    """Bar chart: confusion counts before (GPT-4o) vs after (v2) correction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Combine pairs, Pegon first
    all_pairs = pegon_keys + arabic_keys
    labels = []
    gpt_vals = []
    v2_vals = []
    is_pegon = []
    
    for pair in all_pairs:
        g = gpt4o_counts.get(pair, 0)
        v = v2_counts.get(pair, 0)
        if g > 0 or v > 0:  # only show pairs that actually occur
            labels.append(f"{pair[0]}→{pair[1]}")
            gpt_vals.append(g)
            v2_vals.append(v)
            is_pegon.append(pair in dict([(k,v) for k,v in zip(pegon_keys, range(len(pegon_keys)))]))
    
    if not labels:
        print("  No confusion pairs found. Skipping chart.")
        return
    
    n = len(labels)
    x = np.arange(n)
    bar_w = 0.35
    
    fig, ax = plt.subplots(figsize=(max(7, n * 0.8), 4))
    
    bars1 = ax.bar(x - bar_w/2, gpt_vals, bar_w, label='GPT-4o (before)',
                   color='#94A3B8', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + bar_w/2, v2_vals, bar_w, label='Ours v2 (after)',
                   color='#065A82', edgecolor='white', linewidth=0.5)
    
    # Value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    str(int(h)), ha='center', va='bottom', fontsize=7, color='#64748B')
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    str(int(h)), ha='center', va='bottom', fontsize=7, color='#065A82')
    
    # Highlight Pegon region
    n_pegon = sum(1 for p in labels if any(c in p for c in "ڤڠڮڄڽ"))
    if n_pegon > 0:
        ax.axvspan(-0.5, n_pegon - 0.5, alpha=0.06, color='#DC2626', zorder=0)
        ax.text(n_pegon/2 - 0.5, ax.get_ylim()[1] * 0.95, 'Pegon pairs',
                ha='center', fontsize=8, color='#DC2626', style='italic')
    
    ax.set_xticks(x)
    try:
        ax.set_xticklabels(labels, fontsize=8, fontfamily='Noto Sans Arabic', rotation=45, ha='right')
    except:
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    
    ax.set_ylabel('Substitution Count', fontsize=10, fontfamily='serif')
    ax.legend(fontsize=9, loc='upper right')
    
    ax.yaxis.grid(True, color='#E2E8F0', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = os.path.join(OUTPUT_DIR, f"confusion_before_after.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {path}")
    
    plt.close()


if __name__ == '__main__':
    main()