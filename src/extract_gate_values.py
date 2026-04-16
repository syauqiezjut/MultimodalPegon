"""
=============================================================================
Extract Gate Values (α) During Inference
=============================================================================
Visualizes how the learned gate opens/closes per byte position.
Shows that gate opens at Pegon-specific character positions.

Run on AutoDL:
    python extract_gate_values.py

Requires: trained v2 checkpoint, test data, model code
Output:  gate_analysis/ folder with plots + JSON
=============================================================================
"""

import os
import sys
import json
import torch
import numpy as np
import unicodedata

# ── Paths ──
BASE = "/root/autodl-tmp/castle-pegon"
sys.path.insert(0, os.path.join(BASE, "src"))

CHECKPOINT_V2 = os.path.join(BASE, "results/step1v2_finetuned_vit/checkpoint-best")
TEST_JSON = os.path.join(BASE, "data/splits/test.json")  # adjust if needed
GPT4O_JSON = os.path.join(BASE, "results/ocr/tier3_gpt_gpt_4o_predictions.json")
IMAGE_DIR = os.path.join(BASE, "data/processed/crops")
OUTPUT_DIR = os.path.join(BASE, "results/gate_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Pegon characters ──
PEGON_SPECIFIC = set("ڤڠڮڄڽۑ")
ARABIC_BASES = {'ڤ': 'ف', 'ڠ': 'غ', 'ڮ': 'ك', 'ڄ': 'ج', 'ڽ': 'ن'}


def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\u0640", "")
    return " ".join(text.split()).strip()


def load_model_and_tokenizer():
    """Load the v2 model with gate value extraction."""
    from transformers import AutoTokenizer
    
    # Import your model class
    try:
        from model_step1 import VisualTextFusionModel
    except ImportError:
        print("ERROR: Cannot import VisualTextFusionModel.")
        print("Make sure src/model_step1.py is in the path.")
        print(f"Current sys.path includes: {BASE}/src")
        sys.exit(1)
    
    # Local paths (AutoDL has no internet)
    LOCAL_BYT5 = os.path.join(BASE, "models/byt5-base")
    LOCAL_VIT = os.path.join(BASE, "models/vit-base-patch16-224")
    
    # If ViT path doesn't exist, try alternatives
    VIT_CANDIDATES = [
        os.path.join(BASE, "models/vit-base-patch16-224"),
        os.path.join(BASE, "models/vit-base"),
        os.path.join(BASE, "models/google-vit-base-patch16-224"),
    ]
    for vp in VIT_CANDIDATES:
        if os.path.isdir(vp):
            LOCAL_VIT = vp
            break
    
    print(f"  ByT5 path: {LOCAL_BYT5}")
    print(f"  ViT path:  {LOCAL_VIT}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_BYT5)
    
    # Instantiate model (constructor loads pretrained ByT5 + ViT)
    model = VisualTextFusionModel(
        byt5_path=LOCAL_BYT5,
        vit_path=LOCAL_VIT,
        freeze_vit=False,  # will load trained weights anyway
        dropout=0.1,
    )
    
    # Load trained checkpoint (state_dict)
    ckpt_path = os.path.join(CHECKPOINT_V2, "model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: Checkpoint not found: {ckpt_path}")
        print(f"  Trying alternatives...")
        for alt in ["model.pt", "pytorch_model.bin", "checkpoint.pt"]:
            alt_path = os.path.join(CHECKPOINT_V2, alt)
            if os.path.exists(alt_path):
                ckpt_path = alt_path
                break
        else:
            # Maybe CHECKPOINT_V2 itself is the .pt file
            if os.path.isfile(CHECKPOINT_V2):
                ckpt_path = CHECKPOINT_V2
    
    print(f"  Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    
    print(f"  Model loaded successfully!")
    return model, tokenizer


def extract_gate_for_sample(model, tokenizer, image_path, ocr_text, max_length=256):
    """
    Run forward pass on one sample and extract gate values.
    
    The gate is model.gate = nn.Sequential(Linear, Sigmoid).
    We hook into the Sigmoid to capture alpha values.
    """
    from PIL import Image
    from torchvision import transforms
    
    # Prepare image (same as training: pad to square, resize 224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    max_side = max(w, h)
    padded = Image.new('RGB', (max_side, max_side), (255, 255, 255))
    padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    pixel_values = transform(padded).unsqueeze(0).cuda()
    
    # Prepare text input
    inputs = tokenizer(ocr_text, return_tensors="pt", max_length=max_length,
                       truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    
    # We need decoder input (labels) for forward pass
    # Use input_ids as dummy labels just to run forward
    labels = input_ids.clone()
    
    # ── Hook into the gate's Sigmoid layer ──
    # model.gate = nn.Sequential(Linear(2*d, d), Sigmoid())
    # The Sigmoid is model.gate[1]
    gate_values_captured = []
    
    def sigmoid_hook(module, input, output):
        # output is alpha [B, seq, d_model] after sigmoid
        gate_values_captured.append(output.detach().cpu())
    
    # Register hook on Sigmoid inside gate
    hook = model.gate[1].register_forward_hook(sigmoid_hook)
    
    # ── Run forward pass ──
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
    
    hook.remove()
    
    # ── Also generate prediction text ──
    with torch.no_grad():
        # Use ByT5's generate through the model
        # Encode first
        encoder_outputs = model.byt5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_t = encoder_outputs.last_hidden_state
        
        # Visual encoding
        vit_outputs = model.vit(pixel_values=pixel_values)
        h_v = vit_outputs.last_hidden_state
        h_v_proj = model.visual_proj(h_v)
        
        # Cross attention
        h_cross, _ = model.cross_attn(h_t, h_v_proj, h_v_proj)
        h_cross = model.cross_attn_norm(h_cross)
        
        # Gate
        gate_input = torch.cat([h_t, h_cross], dim=-1)
        alpha = model.gate(gate_input)
        h_fused = alpha * h_cross + (1 - alpha) * h_t
        h_fused = model.fusion_norm(h_fused)
        
        # Generate using ByT5 decoder with fused encoder output
        from transformers.modeling_outputs import BaseModelOutput
        fused_encoder_out = BaseModelOutput(last_hidden_state=h_fused)
        
        gen_ids = model.byt5.generate(
            encoder_outputs=fused_encoder_out,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=1,
        )
        prediction = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    
    # ── Extract gate values ──
    if gate_values_captured:
        gate_raw = gate_values_captured[0]  # [B, seq, d_model]
        # Mean over hidden dim to get per-position gate value
        gate_mean = gate_raw[0].mean(dim=-1).numpy()  # [seq]
    else:
        gate_mean = None
        print("  WARNING: No gate values captured.")
    
    # Get byte-level tokens (only non-padding)
    seq_len = attention_mask[0].sum().item()
    byte_ids = input_ids[0, :seq_len].cpu().tolist()
    byte_chars = []
    for bid in byte_ids:
        if bid < 3:  # special tokens (pad=0, eos=1, unk=2)
            byte_chars.append('<S>')
        else:
            try:
                byte_chars.append(chr(bid - 3))
            except:
                byte_chars.append('?')
    
    if gate_mean is not None:
        gate_mean = gate_mean[:seq_len]
    
    return gate_mean, byte_chars, prediction, seq_len


def map_bytes_to_chars(byte_chars, gate_values):
    """
    Aggregate byte-level gate values to character level.
    UTF-8 Arabic characters are typically 2 bytes.
    """
    chars = []
    char_gates = []
    
    i = 0
    current_char_bytes = []
    current_gate_vals = []
    
    for i, (bc, gv) in enumerate(zip(byte_chars, gate_values)):
        if bc == '<S>':
            continue
        current_char_bytes.append(bc)
        current_gate_vals.append(gv)
        
        # Try to decode accumulated bytes
        try:
            raw = ''.join(current_char_bytes)
            raw.encode('utf-8')  # test if valid
            chars.append(raw)
            char_gates.append(np.mean(current_gate_vals))
            current_char_bytes = []
            current_gate_vals = []
        except:
            continue
    
    return chars, char_gates


def plot_gate_values(chars, gate_values, record_id, ocr_text, ref_text, output_path):
    """Plot gate values per character with Pegon highlighting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n = len(chars)
    if n == 0:
        print(f"  No characters to plot for {record_id}")
        return
    
    fig, ax = plt.subplots(figsize=(max(8, n * 0.25), 3.5))
    
    x = np.arange(n)
    colors = []
    for c in chars:
        if c in PEGON_SPECIFIC:
            colors.append('#DC2626')  # red for Pegon
        elif c in ARABIC_BASES.values():
            colors.append('#F59E0B')  # amber for Arabic bases
        elif c == ' ':
            colors.append('#E2E8F0')  # light for spaces
        else:
            colors.append('#065A82')  # blue for standard
    
    bars = ax.bar(x, gate_values, color=colors, edgecolor='white', linewidth=0.3, width=0.8)
    
    # Character labels on x-axis
    try:
        ax.set_xticks(x)
        ax.set_xticklabels(chars, fontsize=7, fontfamily='Noto Sans Arabic', rotation=0)
    except:
        ax.set_xticks(x)
        ax.set_xticklabels(chars, fontsize=7, rotation=0)
    
    ax.set_ylabel('Gate value α', fontsize=10, fontfamily='serif')
    ax.set_ylim(0, max(gate_values) * 1.3 + 0.01)
    
    # Grid
    ax.yaxis.grid(True, color='#E2E8F0', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DC2626', label='Pegon-specific char'),
        Patch(facecolor='#F59E0B', label='Arabic base (confusable)'),
        Patch(facecolor='#065A82', label='Standard character'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')
    
    # Title
    ax.set_title(f'{record_id}', fontsize=9, fontfamily='sans-serif', color='#64748B')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 65)
    print("  Gate Value Extraction & Visualization")
    print("=" * 65)
    
    # Load GPT-4o predictions (to get OCR text + ground truth)
    with open(GPT4O_JSON, 'r', encoding='utf-8') as f:
        gpt4o_data = json.load(f)
    gpt4o_dict = {r['record_id']: r for r in gpt4o_data}
    
    # Load model
    print("\nLoading model...")
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        print("\n── ALTERNATIVE: Manual gate extraction ──")
        print("If your model doesn't support hooks easily,")
        print("add this to model_step1.py's forward():")
        print("  self.last_gate_values = alpha  # after sigmoid")
        print("Then access: model.last_gate_values after forward()")
        return
    
    # Pick samples to analyze (diverse kitabs)
    # Select a few interesting ones from each kitab
    target_kitabs = ['Kitab01', 'Kitab02', 'Kitab03', 'Kitab04', 'Kitab05']
    selected = []
    for kid in target_kitabs:
        candidates = [r for r in gpt4o_data if r.get('kitab_id') == kid]
        if candidates:
            # Pick one with moderate CER (not too easy, not too hard)
            selected.append(candidates[len(candidates)//3])
    
    print(f"\nAnalyzing {len(selected)} samples...")
    
    all_results = []
    for sample in selected:
        rid = sample['record_id']
        kitab = sample.get('kitab_id', rid.split('_')[0])
        ocr_text = sample.get('gpt_gpt_4o_norm') or sample.get('gpt_gpt_4o', '')
        ref_text = sample.get('gt_normalized') or sample.get('ground_truth', '')
        
        # Find image
        img_path = os.path.join(IMAGE_DIR, kitab, f"{rid}.png")
        if not os.path.exists(img_path):
            print(f"  Image not found: {img_path}")
            continue
        
        print(f"\n  Processing: {rid} ({kitab})")
        print(f"  OCR: {ocr_text[:60]}...")
        print(f"  Ref: {ref_text[:60]}...")
        
        gate_values, byte_chars, prediction, seq_len = extract_gate_for_sample(
            model, tokenizer, img_path, ocr_text)
        
        if gate_values is not None:
            # Aggregate to character level
            chars, char_gates = map_bytes_to_chars(byte_chars, gate_values)
            
            # Plot
            plot_path = os.path.join(OUTPUT_DIR, f"gate_{rid}.png")
            plot_gate_values(chars, char_gates, rid, ocr_text, ref_text, plot_path)
            
            # Save data
            all_results.append({
                'record_id': rid,
                'kitab': kitab,
                'ocr_text': ocr_text,
                'ref_text': ref_text,
                'prediction': prediction,
                'chars': chars,
                'gate_values': [float(g) for g in char_gates],
                'mean_gate': float(np.mean(char_gates)),
                'max_gate': float(np.max(char_gates)),
                'pegon_char_gates': {c: float(g) for c, g in zip(chars, char_gates) if c in PEGON_SPECIFIC},
                'arabic_base_gates': {c: float(g) for c, g in zip(chars, char_gates) if c in ARABIC_BASES.values()},
            })
    
    # Save all results
    json_path = os.path.join(OUTPUT_DIR, "gate_analysis_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {json_path}")
    
    # Summary statistics
    if all_results:
        all_means = [r['mean_gate'] for r in all_results]
        all_maxs = [r['max_gate'] for r in all_results]
        pegon_gates = []
        other_gates = []
        for r in all_results:
            for c, g in zip(r['chars'], r['gate_values']):
                if c in PEGON_SPECIFIC:
                    pegon_gates.append(g)
                elif c != ' ':
                    other_gates.append(g)
        
        print(f"\n  ── Gate Value Summary ──")
        print(f"  Mean α (all chars):     {np.mean(other_gates + pegon_gates):.4f}")
        print(f"  Mean α (Pegon chars):   {np.mean(pegon_gates):.4f}" if pegon_gates else "  No Pegon chars found")
        print(f"  Mean α (other chars):   {np.mean(other_gates):.4f}" if other_gates else "")
        if pegon_gates and other_gates:
            ratio = np.mean(pegon_gates) / max(np.mean(other_gates), 1e-8)
            print(f"  Ratio (Pegon/other):    {ratio:.2f}x")
    
    print(f"\n{'=' * 65}")
    print("  NOTE: If gate hook didn't work, modify model_step1.py:")
    print("  In the forward method, after computing alpha = sigmoid(...):")
    print("    self.last_alpha = alpha.detach()")
    print("  Then access model.last_alpha after forward pass.")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()