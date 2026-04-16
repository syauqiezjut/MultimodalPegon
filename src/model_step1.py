"""
CASTLE-Pegon Step 1: ByT5 + ViT Simple Cross-Attention Fusion

Architecture:
  1. ViT-Base/16 encodes image → h_v [B, 197, 768]
  2. Linear projection: h_v → h_v' [B, 197, d_model]
  3. ByT5 encoder encodes OCR text → h_t [B, seq, d_model]
  4. Cross-attention: Q=h_t, K=h_v', V=h_v' → h_cross [B, seq, d_model]
  5. Gated fusion: h_fused = α * h_cross + (1-α) * h_t  (learnable gate)
  6. ByT5 decoder(h_fused) → corrected text

This is Step 1: NO KG, NO HMG. Just visual + text fusion.
"""

import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    ViTModel,
)


class VisualTextFusionModel(nn.Module):
    """ByT5 + ViT with cross-attention fusion for post-OCR correction."""

    def __init__(
        self,
        byt5_path: str,
        vit_path: str,
        freeze_vit: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- Load pretrained models ---
        self.byt5 = T5ForConditionalGeneration.from_pretrained(byt5_path)
        self.vit = ViTModel.from_pretrained(vit_path)

        self.d_model = self.byt5.config.d_model  # 1536 for ByT5-base
        self.vit_hidden = self.vit.config.hidden_size  # 768 for ViT-Base

        # --- Freeze ViT if specified ---
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

        # --- Projection: ViT hidden → ByT5 d_model ---
        self.visual_proj = nn.Sequential(
            nn.Linear(self.vit_hidden, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )

        # --- Cross-Attention: text attends to visual ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.d_model)

        # --- Simple learnable gate ---
        self.gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid(),
        )

        # --- Initialize gate to strongly favor TEXT (alpha ≈ 0.02) ---
        # This ensures the model starts by mostly copying ByT5 output
        # and gradually learns to incorporate visual features
        nn.init.constant_(self.gate[0].bias, -4.0)

        # --- Output LayerNorm before decoder ---
        self.fusion_norm = nn.LayerNorm(self.d_model)

    def freeze_vit(self):
        """Freeze ViT parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

    def unfreeze_vit(self):
        """Unfreeze ViT parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True
        self.vit.train()

    def forward(
        self,
        input_ids,         # [B, seq] - ByT5 input (OCR text)
        attention_mask,    # [B, seq]
        pixel_values,      # [B, 3, 224, 224] - image
        labels=None,       # [B, tgt_seq] - ground truth for training
    ):
        # --- 1. Encode image with ViT ---
        with torch.no_grad() if not self.vit.training else torch.enable_grad():
            vit_out = self.vit(pixel_values=pixel_values)
        h_v = vit_out.last_hidden_state  # [B, 197, 768]

        # --- 2. Project visual features ---
        h_v_proj = self.visual_proj(h_v)  # [B, 197, d_model]

        # --- 3. Encode text with ByT5 encoder ---
        encoder_outputs = self.byt5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        h_t = encoder_outputs.last_hidden_state  # [B, seq, d_model]

        # --- 4. Cross-attention: text queries, visual keys/values ---
        h_cross, _ = self.cross_attn(
            query=h_t,
            key=h_v_proj,
            value=h_v_proj,
        )
        h_cross = self.cross_attn_norm(h_cross)  # [B, seq, d_model]

        # --- 5. Gated fusion ---
        gate_input = torch.cat([h_t, h_cross], dim=-1)  # [B, seq, 2*d_model]
        alpha = self.gate(gate_input)  # [B, seq, d_model] values in [0,1]
        h_fused = alpha * h_cross + (1 - alpha) * h_t  # [B, seq, d_model]

        # --- 6. Final norm ---
        h_fused = self.fusion_norm(h_fused)  # [B, seq, d_model]

        # --- 7. Decode with ByT5 decoder ---
        # Replace encoder outputs with our fused representation
        encoder_outputs.last_hidden_state = h_fused

        outputs = self.byt5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs  # .loss and .logits

    def generate(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        **generate_kwargs,
    ):
        """Generate corrected text (inference)."""
        # Encode image
        with torch.no_grad():
            vit_out = self.vit(pixel_values=pixel_values)
        h_v = vit_out.last_hidden_state
        h_v_proj = self.visual_proj(h_v)

        # Encode text
        encoder_outputs = self.byt5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        h_t = encoder_outputs.last_hidden_state

        # Cross-attention + gated fusion
        h_cross, _ = self.cross_attn(query=h_t, key=h_v_proj, value=h_v_proj)
        h_cross = self.cross_attn_norm(h_cross)

        gate_input = torch.cat([h_t, h_cross], dim=-1)
        alpha = self.gate(gate_input)
        h_fused = alpha * h_cross + (1 - alpha) * h_t
        h_fused = self.fusion_norm(h_fused)

        encoder_outputs.last_hidden_state = h_fused

        # Generate
        generated = self.byt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        return generated

    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "total_M": f"{total / 1e6:.1f}M",
            "trainable_M": f"{trainable / 1e6:.1f}M",
        }