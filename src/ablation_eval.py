"""
=============================================================================
Ablation Evaluation: Gate & Cross-Attention Variants
=============================================================================
Evaluates the v2 checkpoint under modified forward passes WITHOUT retraining.
This tests architectural contributions by disabling components at eval time.

Run on AutoDL:
    python ablation_eval.py

Variants tested:
  1. v2 (original)          — full model, baseline comparison
  2. no_gate (α=0.5)        — gate replaced with constant 0.5
  3. gate_zero (α=0)        — gate forced to 0 = text-only (no visual)
  4. gate_one (α=1)          — gate forced to 1 = visual-only (no text residual)
  5. no_crossattn            — skip cross-attention, gate gets h_t concatenated with itself
  6. direct_add              — h_fused = h_t + h_cross (no gate, simple addition)

Output: ablation_results.txt + ablation_results.json
=============================================================================
"""

import os
import sys
import json
import torch
import unicodedata
import numpy as np
from collections import defaultdict

# ── Paths ──
BASE = "/root/autodl-tmp/castle-pegon"
sys.path.insert(0, os.path.join(BASE, "src"))

CHECKPOINT_V2 = os.path.join(BASE, "results/step1v2_finetuned_vit/checkpoint-best/model.pt")
LOCAL_BYT5 = os.path.join(BASE, "models/byt5-base")
IMAGE_DIR = os.path.join(BASE, "data/processed/crops")
OUTPUT_DIR = os.path.join(BASE, "results/ablation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ViT path - try multiple locations
VIT_CANDIDATES = [
    os.path.join(BASE, "models/vit-base-patch16-224"),
    os.path.join(BASE, "models/vit-base"),
    os.path.join(BASE, "models/google-vit-base-patch16-224"),
]
LOCAL_VIT = None
for vp in VIT_CANDIDATES:
    if os.path.isdir(vp):
        LOCAL_VIT = vp
        break
if LOCAL_VIT is None:
    print("ERROR: ViT model not found. Check models/ directory.")
    print(f"Tried: {VIT_CANDIDATES}")
    sys.exit(1)

# Test data - adjust paths as needed
TEST_JSON = None
for candidate in [
    os.path.join(BASE, "data/splits/test.json"),
    os.path.join(BASE, "data/processed/test.json"),
    os.path.join(BASE, "results/ocr/tier3_gpt_gpt_4o_predictions.json"),
]:
    if os.path.exists(candidate):
        TEST_JSON = candidate
        break

MAX_LENGTH = 256
BATCH_SIZE = 8  # adjust for GPU memory


def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\u0640", "")
    return " ".join(text.split()).strip()


def compute_cer(pred, ref):
    """Character Error Rate via edit distance."""
    pred = normalize_text(pred)
    ref = normalize_text(ref)
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    m, n = len(pred), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if pred[i-1] == ref[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / n


def compute_wer(pred, ref):
    """Word Error Rate."""
    pred_words = normalize_text(pred).split()
    ref_words = normalize_text(ref).split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    m, n = len(pred_words), len(ref_words)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if pred_words[i-1] == ref_words[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / n


def load_model():
    """Load v2 model from checkpoint."""
    from transformers import AutoTokenizer
    from model_step1 import VisualTextFusionModel

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_BYT5)
    model = VisualTextFusionModel(
        byt5_path=LOCAL_BYT5,
        vit_path=LOCAL_VIT,
        freeze_vit=False,
        dropout=0.1,
    )
    state_dict = torch.load(CHECKPOINT_V2, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model, tokenizer


def load_test_data():
    """Load test data with OCR text and ground truth from GPT-4o predictions."""
    # GPT-4o file has: record_id, kitab_id, gt_normalized, gpt_gpt_4o_norm
    gpt4o_path = os.path.join(BASE, "results/ocr/tier3_gpt_gpt_4o_predictions.json")
    with open(gpt4o_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for r in data:
        rid = r['record_id']
        kitab = r['kitab_id']
        ocr = r['gpt_gpt_4o_norm']
        ref = r['gt_normalized']

        # Find image
        img_path = os.path.join(IMAGE_DIR, kitab, f"{rid}.png")
        if not os.path.exists(img_path):
            for ext in ['jpg', 'jpeg']:
                alt = os.path.join(IMAGE_DIR, kitab, f"{rid}.{ext}")
                if os.path.exists(alt):
                    img_path = alt
                    break

        if ocr and ref:
            samples.append({
                'record_id': rid,
                'kitab': kitab,
                'ocr_text': ocr,
                'ref_text': ref,
                'image_path': img_path,
            })
    return samples


def prepare_image(image_path):
    """Load and preprocess image."""
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
    else:
        # Blank white image as fallback
        img = Image.new('RGB', (224, 224), (255, 255, 255))

    w, h = img.size
    max_side = max(w, h)
    padded = Image.new('RGB', (max_side, max_side), (255, 255, 255))
    padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    return transform(padded)


def generate_with_ablation(model, tokenizer, samples, variant="original"):
    """
    Run generation with a specific ablation variant.

    Variants:
      original     — normal forward
      no_gate      — α = 0.5 constant
      gate_zero    — α = 0 (text only, ignores visual)
      gate_one     — α = 1 (visual only, ignores text residual)
      no_crossattn — skip cross-attention, use h_t directly
      direct_add   — h_fused = h_t + h_cross (no gate weighting)
    """
    from transformers.modeling_outputs import BaseModelOutput

    predictions = []

    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"    [{variant}] {i}/{len(samples)}...")

        # Prepare inputs
        inputs = tokenizer(
            sample['ocr_text'], return_tensors="pt",
            max_length=MAX_LENGTH, truncation=True, padding="max_length"
        )
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        pixel_values = prepare_image(sample['image_path']).unsqueeze(0).cuda()

        with torch.no_grad():
            # 1. Encode text
            encoder_outputs = model.byt5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            h_t = encoder_outputs.last_hidden_state  # [1, seq, d_model]

            # 2. Encode image
            vit_out = model.vit(pixel_values=pixel_values)
            h_v = vit_out.last_hidden_state  # [1, 197, 768]
            h_v_proj = model.visual_proj(h_v)  # [1, 197, d_model]

            # 3. Apply ablation variant
            if variant == "no_crossattn":
                # Skip cross-attention entirely — only text
                h_fused = h_t

            elif variant == "gate_zero":
                # α=0 → h_fused = (1-0)*h_t = h_t (text only)
                h_fused = h_t

            elif variant == "gate_one":
                # α=1 → h_fused = 1*h_cross + 0*h_t
                h_cross, _ = model.cross_attn(h_t, h_v_proj, h_v_proj)
                h_cross = model.cross_attn_norm(h_cross)
                h_fused = model.fusion_norm(h_cross)

            elif variant == "no_gate":
                # Cross-attention + constant α=0.5
                h_cross, _ = model.cross_attn(h_t, h_v_proj, h_v_proj)
                h_cross = model.cross_attn_norm(h_cross)
                h_fused = 0.5 * h_cross + 0.5 * h_t
                h_fused = model.fusion_norm(h_fused)

            elif variant == "direct_add":
                # Cross-attention + simple addition (no gate)
                h_cross, _ = model.cross_attn(h_t, h_v_proj, h_v_proj)
                h_cross = model.cross_attn_norm(h_cross)
                h_fused = h_t + h_cross
                h_fused = model.fusion_norm(h_fused)

            else:  # "original"
                # Normal forward with learned gate
                h_cross, _ = model.cross_attn(h_t, h_v_proj, h_v_proj)
                h_cross = model.cross_attn_norm(h_cross)
                gate_input = torch.cat([h_t, h_cross], dim=-1)
                alpha = model.gate(gate_input)
                h_fused = alpha * h_cross + (1 - alpha) * h_t
                h_fused = model.fusion_norm(h_fused)

            # 4. Decode
            fused_out = BaseModelOutput(last_hidden_state=h_fused)
            gen_ids = model.byt5.generate(
                encoder_outputs=fused_out,
                attention_mask=attention_mask,
                max_length=MAX_LENGTH,
                num_beams=1,
            )
            pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        predictions.append({
            'record_id': sample['record_id'],
            'kitab': sample['kitab'],
            'prediction': pred,
            'ref_text': sample['ref_text'],
            'ocr_text': sample['ocr_text'],
        })

    return predictions


def evaluate_predictions(predictions):
    """Compute corpus-level CER and WER."""
    cers = []
    wers = []
    per_kitab = defaultdict(list)

    for p in predictions:
        cer = compute_cer(p['prediction'], p['ref_text'])
        wer = compute_wer(p['prediction'], p['ref_text'])
        cers.append(cer)
        wers.append(wer)
        per_kitab[p['kitab']].append(cer)

    macro_cer = np.mean(cers)
    macro_wer = np.mean(wers)
    kitab_cers = {k: np.mean(v) for k, v in sorted(per_kitab.items())}

    return {
        'macro_cer': float(macro_cer),
        'macro_wer': float(macro_wer),
        'n_samples': len(predictions),
        'per_kitab_cer': {k: float(v) for k, v in kitab_cers.items()},
    }


def main():
    print("=" * 65)
    print("  Ablation Evaluation: Gate & Cross-Attention Variants")
    print("=" * 65)

    # Load model & data
    print("\n  Loading model...")
    model, tokenizer = load_model()
    print("  Loading test data...")
    samples = load_test_data()
    print(f"  {len(samples)} test samples loaded")

    # Define variants
    variants = [
        ("original",      "v2 full model (learned gate)"),
        ("no_gate",        "v2 − gate (constant α=0.5)"),
        ("direct_add",     "v2 − gate (h_t + h_cross)"),
        ("gate_zero",      "v2 gate=0 (text-only, no visual)"),
        ("no_crossattn",   "v2 − cross-attention (text encoder only)"),
    ]

    results = {}
    for variant_id, variant_name in variants:
        print(f"\n  ── Evaluating: {variant_name} ──")
        preds = generate_with_ablation(model, tokenizer, samples, variant=variant_id)
        metrics = evaluate_predictions(preds)
        results[variant_id] = {
            'name': variant_name,
            **metrics,
        }
        print(f"    CER: {metrics['macro_cer']:.4f}  |  WER: {metrics['macro_wer']:.4f}")
        for k, v in metrics['per_kitab_cer'].items():
            print(f"      {k}: {v:.4f}")

    # Save results
    json_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print comparison table
    print(f"\n{'=' * 65}")
    print(f"  ABLATION RESULTS")
    print(f"{'=' * 65}")
    print(f"  {'Configuration':<40} {'CER':>8} {'WER':>8} {'Δ CER':>8}")
    print(f"  {'-'*64}")

    baseline_cer = results.get('original', {}).get('macro_cer', 0)
    for variant_id, variant_name in variants:
        r = results[variant_id]
        cer = r['macro_cer']
        wer = r['macro_wer']
        if variant_id == 'original':
            delta = '---'
        else:
            d = (cer - baseline_cer) / baseline_cer * 100
            delta = f"{d:+.1f}%"
        print(f"  {variant_name:<40} {cer:>8.4f} {wer:>8.4f} {delta:>8}")

    # Save text report
    report_path = os.path.join(OUTPUT_DIR, "ablation_results.txt")
    with open(report_path, 'w') as f:
        f.write("ABLATION RESULTS\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Configuration':<40} {'CER':>8} {'WER':>8} {'Δ CER':>8}\n")
        f.write("-" * 65 + "\n")
        for variant_id, variant_name in variants:
            r = results[variant_id]
            cer = r['macro_cer']
            wer = r['macro_wer']
            if variant_id == 'original':
                delta = '---'
            else:
                d = (cer - baseline_cer) / baseline_cer * 100
                delta = f"{d:+.1f}%"
            f.write(f"{variant_name:<40} {cer:>8.4f} {wer:>8.4f} {delta:>8}\n")

    print(f"\n  Results: {json_path}")
    print(f"  Report:  {report_path}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()