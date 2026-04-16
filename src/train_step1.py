"""
CASTLE-Pegon Step 1: Train ByT5 + ViT Simple Fusion
====================================================
Gate criterion: CER < 0.41 on test set (must beat ByT5 text-only 0.4134)

Usage:
  cd /root/autodl-tmp/castle-pegon
  python train_step1.py

Training strategy:
  Phase 1 (epochs 1-10):  Freeze ViT, train fusion + decoder
  Phase 2 (epochs 11-20): Unfreeze ViT, end-to-end (lower LR)
  Early stopping: patience=5 on val CER
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, ViTImageProcessor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model_step1 import VisualTextFusionModel
from src.dataset_castle import PegonCorrectionDataset, normalize_pegon

# ============================================
# CONFIG
# ============================================

class Config:
    # Paths
    project_root = "/root/autodl-tmp/castle-pegon"
    byt5_path = "models/byt5-base"
    vit_path = "models/vit-base"
    image_root = "data/processed/crops"
    train_json = "data/splits/train.json"
    val_json = "data/splits/val.json"
    test_json = "data/splits/test.json"
    output_dir = "results/step1_vit_fusion"

    # Model
    max_source_length = 256
    max_target_length = 256
    dropout = 0.1

    # Training Phase 1 (freeze ViT)
    phase1_epochs = 10
    phase1_lr = 1e-4
    phase1_batch_size = 8
    phase1_grad_accum = 2  # effective batch = 16

    # Training Phase 2 (unfreeze ViT)
    phase2_epochs = 10
    phase2_lr = 5e-5       # lower LR for fine-tuning ViT
    phase2_vit_lr = 1e-5   # even lower for ViT params
    phase2_batch_size = 4  # smaller batch (more memory for ViT grads)
    phase2_grad_accum = 4  # effective batch = 16

    # Generation
    gen_max_length = 256
    gen_num_beams = 4
    eval_num_beams = 1  # greedy during training, beam for final test

    # Early stopping
    patience = 5

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = False
    bf16 = True

    # Logging
    log_every = 50  # steps
    eval_every_epoch = True


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def compute_cer(predictions: list, references: list) -> float:
    """Compute Character Error Rate with Pegon normalization."""
    total_chars = 0
    total_errors = 0
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_pegon(pred)
        ref_norm = normalize_pegon(ref)
        # Simple edit distance
        errors = edit_distance(pred_norm, ref_norm)
        total_errors += errors
        total_chars += max(len(ref_norm), 1)
    return total_errors / max(total_chars, 1)


def compute_wer(predictions: list, references: list) -> float:
    """Compute Word Error Rate with Pegon normalization."""
    total_words = 0
    total_errors = 0
    for pred, ref in zip(predictions, references):
        pred_words = normalize_pegon(pred).split()
        ref_words = normalize_pegon(ref).split()
        errors = edit_distance(pred_words, ref_words)
        total_errors += errors
        total_words += max(len(ref_words), 1)
    return total_errors / max(total_words, 1)


def edit_distance(s1, s2) -> int:
    """Standard Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def evaluate(model, dataloader, tokenizer, config, logger, split_name="val", num_beams=None):
    """Run evaluation, return CER/WER and predictions."""
    model.eval()
    all_preds = []
    all_refs = []
    all_record_ids = []

    logger.info(f"Evaluating on {split_name} ({len(dataloader.dataset)} samples)...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            pixel_values = batch["pixel_values"].to(config.device)
            labels = batch["labels"].to(config.device)

            # Generate
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
                beams = num_beams if num_beams is not None else config.eval_num_beams
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_length=config.gen_max_length,
                    num_beams=beams,
                )

            # Decode predictions
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Decode references (replace -100 with pad_token_id)
            label_ids = labels.clone()
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)
            all_record_ids.extend(batch["record_id"])

            if (batch_idx + 1) % 20 == 0:
                logger.info(f"  Eval batch {batch_idx + 1}/{len(dataloader)}")

    # Compute metrics
    cer = compute_cer(all_preds, all_refs)
    wer = compute_wer(all_preds, all_refs)

    logger.info(f"{split_name} Results: CER={cer:.4f}, WER={wer:.4f}")

    # Show a few examples
    for i in range(min(3, len(all_preds))):
        logger.info(f"  Example {i+1}:")
        logger.info(f"    REF:  {all_refs[i][:100]}")
        logger.info(f"    PRED: {all_preds[i][:100]}")

    model.train()
    return {
        "cer": cer,
        "wer": wer,
        "predictions": [
            {"record_id": rid, "prediction": p, "reference": r}
            for rid, p, r in zip(all_record_ids, all_preds, all_refs)
        ],
    }


def train_one_epoch(
    model, dataloader, optimizer, scaler, config, logger, epoch, global_step
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    # Determine grad accumulation steps based on phase
    grad_accum = (
        config.phase1_grad_accum
        if epoch <= config.phase1_epochs
        else config.phase2_grad_accum
    )

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        pixel_values = batch["pixel_values"].to(config.device)
        labels = batch["labels"].to(config.device)

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss / grad_accum

        # Backward
        scaler.scale(loss).backward()

        # Step optimizer every grad_accum batches
        if (batch_idx + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        total_loss += loss.item() * grad_accum
        num_batches += 1

        if (batch_idx + 1) % config.log_every == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | Step: {global_step}"
            )

    # Handle remaining gradients
    if num_batches % grad_accum != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


def main():
    config = Config()
    os.chdir(config.project_root)
    logger = setup_logging(config.output_dir)

    logger.info("=" * 60)
    logger.info("CASTLE-Pegon Step 1: ByT5 + ViT Fusion Training")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"FP16: {config.fp16}")

    # ---- Load tokenizer & processor ----
    logger.info("Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(config.byt5_path)
    vit_processor = ViTImageProcessor.from_pretrained(config.vit_path)

    # ---- Load datasets ----
    logger.info("Loading datasets...")
    train_dataset = PegonCorrectionDataset(
        json_path=config.train_json,
        image_root=config.image_root,
        vit_processor=vit_processor,
        byt5_tokenizer=tokenizer,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
        split="train",
    )
    val_dataset = PegonCorrectionDataset(
        json_path=config.val_json,
        image_root=config.image_root,
        vit_processor=vit_processor,
        byt5_tokenizer=tokenizer,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
        split="val",
    )
    test_dataset = PegonCorrectionDataset(
        json_path=config.test_json,
        image_root=config.image_root,
        vit_processor=vit_processor,
        byt5_tokenizer=tokenizer,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
        split="test",
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ---- Build model ----
    logger.info("Building VisualTextFusionModel (ViT frozen)...")
    model = VisualTextFusionModel(
        byt5_path=config.byt5_path,
        vit_path=config.vit_path,
        freeze_vit=True,
        dropout=config.dropout,
    )
    params = model.count_parameters()
    logger.info(f"Parameters: {params}")
    model = model.to(config.device)

    # ---- Scaler for FP16 ----
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # ============================================
    # PHASE 1: Freeze ViT, train fusion + decoder
    # ============================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 1: Freeze ViT (epochs 1-{})".format(config.phase1_epochs))
    logger.info("=" * 60)

    train_loader_p1 = DataLoader(
        train_dataset,
        batch_size=config.phase1_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.phase1_batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.phase1_batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer: only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_p1 = AdamW(trainable_params, lr=config.phase1_lr, weight_decay=0.01)
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=config.phase1_epochs)

    best_val_cer = float("inf")
    patience_counter = 0
    global_step = 0
    all_metrics = []

    for epoch in range(1, config.phase1_epochs + 1):
        epoch_start = time.time()
        logger.info(f"\n--- Phase 1, Epoch {epoch}/{config.phase1_epochs} ---")

        avg_loss, global_step = train_one_epoch(
            model, train_loader_p1, optimizer_p1, scaler,
            config, logger, epoch, global_step,
        )
        scheduler_p1.step()

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} done in {epoch_time:.0f}s | Avg loss: {avg_loss:.4f}")

        # Evaluate on val
        val_result = evaluate(model, val_loader, tokenizer, config, logger, "val")

        metrics = {
            "epoch": epoch,
            "phase": 1,
            "train_loss": avg_loss,
            "val_cer": val_result["cer"],
            "val_wer": val_result["wer"],
            "lr": optimizer_p1.param_groups[0]["lr"],
            "time_s": epoch_time,
        }
        all_metrics.append(metrics)

        # Early stopping check
        if val_result["cer"] < best_val_cer:
            best_val_cer = val_result["cer"]
            patience_counter = 0
            # Save best model
            save_path = os.path.join(config.output_dir, "checkpoint-best")
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            logger.info(f"New best val CER: {best_val_cer:.4f} → saved to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            logger.info("Early stopping triggered in Phase 1!")
            break

    # ============================================
    # PHASE 2: Unfreeze ViT, end-to-end
    # ============================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2: Unfreeze ViT (epochs {}-{})".format(
        config.phase1_epochs + 1, config.phase1_epochs + config.phase2_epochs
    ))
    logger.info("=" * 60)

    # Load best Phase 1 checkpoint
    best_ckpt = os.path.join(config.output_dir, "checkpoint-best", "model.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=config.device))
        logger.info(f"Loaded best Phase 1 checkpoint (val CER: {best_val_cer:.4f})")

    # Unfreeze ViT
    model.unfreeze_vit()
    params = model.count_parameters()
    logger.info(f"After unfreezing ViT: {params}")

    train_loader_p2 = DataLoader(
        train_dataset,
        batch_size=config.phase2_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Separate param groups: ViT gets lower LR
    vit_params = [p for n, p in model.named_parameters() if "vit" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "vit" not in n and p.requires_grad]

    optimizer_p2 = AdamW([
        {"params": other_params, "lr": config.phase2_lr},
        {"params": vit_params, "lr": config.phase2_vit_lr},
    ], weight_decay=0.01)
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=config.phase2_epochs)

    patience_counter = 0  # reset

    for epoch in range(config.phase1_epochs + 1, config.phase1_epochs + config.phase2_epochs + 1):
        epoch_start = time.time()
        logger.info(f"\n--- Phase 2, Epoch {epoch} ---")

        avg_loss, global_step = train_one_epoch(
            model, train_loader_p2, optimizer_p2, scaler,
            config, logger, epoch, global_step,
        )
        scheduler_p2.step()

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} done in {epoch_time:.0f}s | Avg loss: {avg_loss:.4f}")

        # Evaluate
        val_result = evaluate(model, val_loader, tokenizer, config, logger, "val")

        metrics = {
            "epoch": epoch,
            "phase": 2,
            "train_loss": avg_loss,
            "val_cer": val_result["cer"],
            "val_wer": val_result["wer"],
            "lr": optimizer_p2.param_groups[0]["lr"],
            "time_s": epoch_time,
        }
        all_metrics.append(metrics)

        if val_result["cer"] < best_val_cer:
            best_val_cer = val_result["cer"]
            patience_counter = 0
            save_path = os.path.join(config.output_dir, "checkpoint-best")
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            logger.info(f"New best val CER: {best_val_cer:.4f} → saved")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            logger.info("Early stopping triggered in Phase 2!")
            break

    # ============================================
    # FINAL EVALUATION ON TEST SET
    # ============================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)

    # Load best model
    model.load_state_dict(torch.load(
        os.path.join(config.output_dir, "checkpoint-best", "model.pt"),
        map_location=config.device,
    ))

    test_result = evaluate(model, test_loader, tokenizer, config, logger, "test", num_beams=config.gen_num_beams)

    # ============================================
    # SAVE ALL RESULTS
    # ============================================
    results = {
        "step": "step1_vit_fusion",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "phase1_epochs": config.phase1_epochs,
            "phase2_epochs": config.phase2_epochs,
            "phase1_lr": config.phase1_lr,
            "phase2_lr": config.phase2_lr,
            "dropout": config.dropout,
            "max_source_length": config.max_source_length,
        },
        "best_val_cer": best_val_cer,
        "test_cer": test_result["cer"],
        "test_wer": test_result["wer"],
        "gate_criterion": "CER < 0.41 (must beat ByT5 text-only 0.4134)",
        "gate_passed": test_result["cer"] < 0.41,
        "training_metrics": all_metrics,
    }

    # Save results
    with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save predictions
    with open(os.path.join(config.output_dir, "predictions_test.json"), "w") as f:
        json.dump(test_result["predictions"], f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 1 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best val CER:  {best_val_cer:.4f}")
    logger.info(f"Test CER:      {test_result['cer']:.4f}")
    logger.info(f"Test WER:      {test_result['wer']:.4f}")
    logger.info(f"ByT5 baseline: 0.4134")
    logger.info(f"GPT-4o:        0.4242")
    logger.info(f"Gate passed:   {results['gate_passed']}")
    logger.info(f"Results saved: {config.output_dir}")
    logger.info("=" * 60)

    if not results["gate_passed"]:
        logger.warning("")
        logger.warning("⚠ GATE NOT PASSED: Visual channel did not improve over text-only.")
        logger.warning("  Consider: check image loading, fusion mechanism, or pivot to Option A/B.")
    else:
        logger.info("")
        logger.info("✓ GATE PASSED: Visual fusion improves over text-only! Proceed to Step 2 (PegonKG).")


if __name__ == "__main__":
    main()