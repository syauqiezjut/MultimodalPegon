"""
=============================================================================
Quantify Lexical vs Glyph Errors
=============================================================================
Uses existing alignment data to classify substitution errors into:
  - Glyph errors: tracked confusable pairs (Pegon + Arabic dot pairs)
  - Lexical/other errors: all remaining substitutions

Run on AutoDL:
    python quantify_error_types.py

Output: error_type_breakdown.txt + .json
=============================================================================
"""

import json
import os
import unicodedata
from collections import Counter

# ── Paths ──
BASE = "/root/autodl-tmp/castle-pegon"
GPT4O_PATH = f"{BASE}/results/ocr/tier3_gpt_gpt_4o_predictions.json"
V2_PATH = f"{BASE}/results/step1v2_finetuned_vit/predictions_test.json"
OUTPUT_DIR = f"{BASE}/results/error_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── All tracked confusable pairs (bidirectional) ──
PEGON_PAIRS = [
    ('ف', 'ڤ'), ('ج', 'ڄ'), ('غ', 'ڠ'), ('ك', 'ڮ'), ('ن', 'ڽ'),
]
ARABIC_DOT_PAIRS = [
    ('ط', 'ظ'), ('ص', 'ض'), ('ع', 'غ'), ('ح', 'خ'), ('د', 'ذ'),
    ('ر', 'ز'), ('ب', 'ت'), ('ت', 'ث'), ('س', 'ش'), ('ج', 'ح'),
]

# Build set of all tracked pairs (both directions)
TRACKED_SET = set()
for a, b in PEGON_PAIRS + ARABIC_DOT_PAIRS:
    TRACKED_SET.add((a, b))
    TRACKED_SET.add((b, a))

PEGON_SET = set()
for a, b in PEGON_PAIRS:
    PEGON_SET.add((a, b))
    PEGON_SET.add((b, a))

ARABIC_DOT_SET = set()
for a, b in ARABIC_DOT_PAIRS:
    ARABIC_DOT_SET.add((a, b))
    ARABIC_DOT_SET.add((b, a))


def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\u0640", "")
    return " ".join(text.split()).strip()


def get_edit_ops(pred_text, ref_text):
    """
    Align pred and ref using edit distance, return operations:
    substitutions, insertions, deletions
    """
    pred = normalize_text(pred_text)
    ref = normalize_text(ref_text)
    m, n = len(pred), len(ref)

    # DP
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
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrack
    substitutions = []
    insertions = 0
    deletions = 0
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred[i-1] == ref[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions.append((pred[i-1], ref[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            insertions += 1
            j -= 1
        else:
            break

    return substitutions, insertions, deletions


def classify_substitution(pred_char, ref_char):
    """Classify a substitution error."""
    pair = (pred_char, ref_char)
    if pair in PEGON_SET:
        return 'pegon_glyph'
    elif pair in ARABIC_DOT_SET:
        return 'arabic_dot_glyph'
    else:
        return 'other'


def main():
    print("=" * 65)
    print("  Quantify Lexical vs Glyph Errors")
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

    # ── Analyze GPT-4o errors ──
    gpt_stats = {'pegon_glyph': 0, 'arabic_dot_glyph': 0, 'other': 0,
                 'insertions': 0, 'deletions': 0, 'total_errors': 0}
    v2_stats = {'pegon_glyph': 0, 'arabic_dot_glyph': 0, 'other': 0,
                'insertions': 0, 'deletions': 0, 'total_errors': 0}

    gpt_sub_examples = Counter()  # top "other" substitution pairs
    v2_sub_examples = Counter()

    for rid in common_ids:
        gpt = gpt4o_dict[rid]
        v2 = v2_dict[rid]
        # GPT-4o file: gt_normalized, gpt_gpt_4o_norm
        # v2 file: reference, prediction
        ref = gpt['gt_normalized']
        gpt_text = gpt['gpt_gpt_4o_norm']
        v2_text = v2['prediction']

        # GPT-4o
        subs, ins, dels = get_edit_ops(gpt_text, ref)
        gpt_stats['insertions'] += ins
        gpt_stats['deletions'] += dels
        gpt_stats['total_errors'] += len(subs) + ins + dels
        for pc, rc in subs:
            cat = classify_substitution(pc, rc)
            gpt_stats[cat] += 1
            if cat == 'other':
                gpt_sub_examples[(pc, rc)] += 1

        # v2
        subs, ins, dels = get_edit_ops(v2_text, ref)
        v2_stats['insertions'] += ins
        v2_stats['deletions'] += dels
        v2_stats['total_errors'] += len(subs) + ins + dels
        for pc, rc in subs:
            cat = classify_substitution(pc, rc)
            v2_stats[cat] += 1
            if cat == 'other':
                v2_sub_examples[(pc, rc)] += 1

    # ── Compute percentages ──
    def pct(count, total):
        return f"{count/max(total,1)*100:.1f}%"

    gpt_total_subs = gpt_stats['pegon_glyph'] + gpt_stats['arabic_dot_glyph'] + gpt_stats['other']
    v2_total_subs = v2_stats['pegon_glyph'] + v2_stats['arabic_dot_glyph'] + v2_stats['other']

    # ── Print results ──
    print(f"\n  {'='*60}")
    print(f"  ERROR TYPE BREAKDOWN")
    print(f"  {'='*60}")

    for label, stats, total_subs in [
        ("GPT-4o (before)", gpt_stats, gpt_total_subs),
        ("v2 (after)", v2_stats, v2_total_subs),
    ]:
        print(f"\n  ── {label} ──")
        print(f"  Total edit operations:       {stats['total_errors']}")
        print(f"    Substitutions:             {total_subs}")
        print(f"      Pegon glyph pairs:       {stats['pegon_glyph']:>5} ({pct(stats['pegon_glyph'], total_subs)})")
        print(f"      Arabic dot pairs:        {stats['arabic_dot_glyph']:>5} ({pct(stats['arabic_dot_glyph'], total_subs)})")
        glyph_total = stats['pegon_glyph'] + stats['arabic_dot_glyph']
        print(f"      ── Total glyph:          {glyph_total:>5} ({pct(glyph_total, total_subs)})")
        print(f"      Other (lexical/context):  {stats['other']:>5} ({pct(stats['other'], total_subs)})")
        print(f"    Insertions:                {stats['insertions']}")
        print(f"    Deletions:                 {stats['deletions']}")

    # ── Improvement per category ──
    print(f"\n  ── Improvement by Error Type ──")
    for cat in ['pegon_glyph', 'arabic_dot_glyph', 'other']:
        g = gpt_stats[cat]
        v = v2_stats[cat]
        delta = g - v
        red = (1 - v / max(g, 1)) * 100 if g > 0 else 0
        print(f"    {cat:<25} {g:>5} → {v:>5}  (Δ={delta:>+4}, {red:>+.1f}%)")

    for cat in ['insertions', 'deletions']:
        g = gpt_stats[cat]
        v = v2_stats[cat]
        delta = g - v
        red = (1 - v / max(g, 1)) * 100 if g > 0 else 0
        print(f"    {cat:<25} {g:>5} → {v:>5}  (Δ={delta:>+4}, {red:>+.1f}%)")

    # ── Top "other" substitution pairs ──
    print(f"\n  ── Top 15 Non-Glyph Substitution Pairs (v2) ──")
    for (pc, rc), cnt in v2_sub_examples.most_common(15):
        print(f"    '{pc}' → '{rc}' : {cnt}")

    # ── Save ──
    output = {
        'gpt4o': {
            'total_subs': gpt_total_subs,
            'pegon_glyph': gpt_stats['pegon_glyph'],
            'arabic_dot_glyph': gpt_stats['arabic_dot_glyph'],
            'glyph_total': gpt_stats['pegon_glyph'] + gpt_stats['arabic_dot_glyph'],
            'other_subs': gpt_stats['other'],
            'insertions': gpt_stats['insertions'],
            'deletions': gpt_stats['deletions'],
            'total_errors': gpt_stats['total_errors'],
        },
        'v2': {
            'total_subs': v2_total_subs,
            'pegon_glyph': v2_stats['pegon_glyph'],
            'arabic_dot_glyph': v2_stats['arabic_dot_glyph'],
            'glyph_total': v2_stats['pegon_glyph'] + v2_stats['arabic_dot_glyph'],
            'other_subs': v2_stats['other'],
            'insertions': v2_stats['insertions'],
            'deletions': v2_stats['deletions'],
            'total_errors': v2_stats['total_errors'],
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "error_type_breakdown.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ── LaTeX snippet ──
    v2_glyph = v2_stats['pegon_glyph'] + v2_stats['arabic_dot_glyph']
    v2_other = v2_stats['other']
    v2_ins = v2_stats['insertions']
    v2_del = v2_stats['deletions']
    v2_total = v2_stats['total_errors']

    print(f"\n  ── LaTeX snippet for paper ──")
    print(f"""
Of the {v2_total:,} total edit operations in the v2 output,
{v2_total_subs:,} are substitutions, {v2_ins:,} are insertions,
and {v2_del:,} are deletions. Among substitutions, {v2_glyph}
({v2_glyph/max(v2_total_subs,1)*100:.1f}\\%) involve tracked
glyph-confusable pairs (Table~\\ref{{tab:char_confusion}}),
while the remaining {v2_other} ({v2_other/max(v2_total_subs,1)*100:.1f}\\%)
are lexical or contextual substitutions outside the known
confusion pairs. This indicates that the majority of remaining
errors are not addressable through improved visual disambiguation
alone and would require lexical constraints or language modeling.
""")

    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()