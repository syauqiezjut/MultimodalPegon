# PegonKitab — Dataset and Code for Multimodal Post-OCR Correction of Pegon Manuscripts

This repository contains the dataset splits, code, and evaluation outputs for the paper:

> **When Arabic OCR Fails: Benchmarking and Multimodal Post-OCR Correction for Pegon Manuscripts**
> Syauqie Muhammad Marier, Bayu Nadya Kusuma, Md Mehedi Hassan Dorjoy, Linan Zhu, Xiangjie Kong
> *Pattern Recognition* (under review, PR-D-26-05439)

---

## Overview

**Pegon** is a modified Arabic script used in Javanese and Sundanese Islamic manuscripts. Several Pegon-specific characters differ from standard Arabic letters by only one or two dots, so OCR systems trained on Arabic systematically confuse them.

This work introduces:
- **PegonKitab** — 2,801 annotated handwritten line images from five manuscripts
- **A 16-system OCR benchmark** spanning traditional OCR, transformer HTR, vision–language models, Arabic-specialized systems, and text-only correction
- **A multimodal post-OCR correction model** (ByT5 + ViT with gated cross-attention) that reduces CER by 18.6% relative to GPT-4o

---

## Dataset

### PegonKitab statistics

Splits are page-level: all lines from the same page belong to the same split.

| Kitab | Language | Train | Val | Test | Total |
|-------|----------|------:|----:|-----:|------:|
| Kitab01 | Javanese  | 168 | 30 | 39 | 237 |
| Kitab02 | Javanese  | 211 | 29 | 84 | 324 |
| Kitab03 | Sundanese | 107 | 12 | 36 | 155 |
| Kitab04 | Sundanese | 476 | 96 | 118 | 690 |
| Kitab05 | Javanese  | 974 | 192 | 229 | 1,395 |
| **Total** | | **1,936** | **359** | **506** | **2,801** |

### Download

| Resource | Location |
|----------|----------|
| Annotation splits (JSON) | this repository, `data/splits/` |
| Cropped line images | [Google Drive](https://drive.google.com/drive/folders/1rtQpl-DuXsU3idGxg_jB9u5GrkWTlq8s?usp=share_link) |

Place the cropped images under `data/processed/crops/<kitab_id>/` so that each record's `cropped_image_path` resolves relative to `data/processed/`.

### Record format

Each file in `data/splits/` is a list of records:

| Field | Description |
|-------|-------------|
| `record_id` | Unique line identifier (`<kitab>_p<page>_<line_id>`) |
| `kitab_id` | Source manuscript (`Kitab01`–`Kitab05`) |
| `page_number`, `image_filename` | Source page |
| `line_id`, `line_order`, `bbox` | Label Studio line ID, reading order, and bounding box (percent of page size) |
| `original_width`, `original_height` | Page size in pixels |
| `ground_truth` | Human transcription in Pegon |
| `language` | Language label from the annotation metadata |
| `cropped_image_path` | Relative path of the cropped line image |
| `split` | `train`, `val`, or `test` |
| `ocr_gpt4o` | GPT-4o zero-shot OCR output (input to the correction models) |

---

## OCR benchmark results

PegonKitab test set (506 lines). CER and WER are macro-averaged after NFC normalization and removal of ZWNJ, ZWJ, and tatweel. RC = recognition coverage (% of inputs with non-empty output).

| Tier | Model | RC (%) | CER | WER |
|------|-------|-------:|----:|----:|
| 1 | Tesseract 5 | 65.2 | 0.7067 | 1.0348 |
| 1 | EasyOCR | 100.0 | 0.6194 | 1.1256 |
| 2 | TrOCR-small | 100.0 | 1.1649 | 1.4599 |
| 2 | TrOCR-base | 100.0 | 1.0196 | 1.2140 |
| 2 | TrOCR-large | 100.0 | 0.9712 | 1.0654 |
| 2 | Arabic-Nougat | 0.0 | 1.0000 | 1.0000 |
| 2 | GOT-OCR2 | 0.0 | 1.0000 | 1.0000 |
| 3 | **GPT-4o** | 100.0 | **0.4243** | **0.9986** |
| 3 | GPT-4o-mini | 100.0 | 0.6628 | 1.1001 |
| 3 | Qwen2.5-VL-3B | 100.0 | 0.5299 | 1.1211 |
| 3 | Qwen2.5-VL-7B | 100.0 | 0.5240 | 1.0585 |
| 3 | InternVL2.5-4B | 100.0 | 1.2431 | 1.7034 |
| 4 | QARI-OCR v0.2 | 98.6 | 1.0364 | 1.7120 |
| 4 | Arabic-HTR-v2 | 100.0 | 27.561 | 11.193 |
| 5 | ByT5 (random aug.) | 100.0 | 0.4134 | 0.9834 |
| 5 | ByT5 (glyph aug.) | 100.0 | 0.4816 | 0.9869 |

Per-system metrics and predictions are in `results/ocr/benchmark/` (GPT-4o predictions: `results/ocr/tier3_gpt_gpt_4o_predictions.json`).

---

## Multimodal correction model

### Architecture

- **Text encoder**: ByT5-base encoder over the OCR transcript (byte-level, no vocabulary changes for Pegon letters)
- **Visual encoder**: ViT-Base/16 over the line image, projected to the ByT5 hidden size
- **Fusion**: 8-head cross-attention (text queries, image keys/values) followed by a learned gate, α = σ(W[h_t; h_cross] + b), with the bias initialized to −4
- **Decoder**: ByT5-base decoder
- **Training**: two phases — frozen ViT, then end-to-end fine-tuning with a lower ViT learning rate

### Results

| Model | CER | WER | Δ CER vs GPT-4o |
|-------|----:|----:|----------------:|
| GPT-4o (zero-shot) | 0.4243 | 0.9986 | — |
| ByT5 text-only | 0.4134 | 0.9834 | −2.6% |
| ByT5-base + ViT (v1) | 0.3635 | 0.8661 | −14.3% |
| **ByT5-ft + ViT (v2)** | **0.3456** | **0.8306** | **−18.6%** |

Test predictions for v1 and v2 are in `results/ocr/step1_vit_fusion/` and `results/ocr/step1v2_finetuned_vit/`. Model checkpoints are not released.

---

## Repository layout

```
config.py                         paths (override with PEGON_BASE_DIR, PEGON_KITAB_DIR, PEGON_ENV)
data/splits/                      train / val / test annotation splits
data/confusion_matrix/            glyph similarity matrix used for augmentation
data/confusion_matrix/table3_version/  matrix behind Table 3 of the paper
src/
  phase01_parse.py … phase04_split.py   build the dataset from Label Studio exports
  phase05_ocr_tier*.py, phase05_ocr_gpt4o_*.py   OCR benchmark (Tiers 1–4)
  glyph_confusion_table3.py, phase06_glyph_confusion.py   glyph similarity matrix
  phase07_augment.py, phase08_byt5.py   text-only correction (Tier 5)
  model_step1.py, dataset_castle.py, train_step1.py, train_step1_v2.py   multimodal model (v1, v2)
  ablation_eval.py, extract_gate_values.py   ablation and gate analysis
  char_confusion_rates.py, quantify_error_types.py   error analysis
results/                          benchmark outputs, predictions, ablation, error analysis
analysis/figures/                 paper figures
```

The training and analysis scripts were run on AutoDL; their configuration sections still contain the original absolute paths (`/root/autodl-tmp/castle-pegon`), which should be edited before running. OpenAI-based scripts read the key from the `OPENAI_API_KEY` environment variable.

### Paper tables and their sources

| Paper | Script | Output |
|-------|--------|--------|
| Table 2 | `phase01`–`phase04` | `data/splits/` |
| Table 3, Fig. 2 | `glyph_confusion_table3.py` | `data/confusion_matrix/table3_version/` |
| Table 5 | `phase05_ocr_*`, `phase08_byt5.py` | `results/ocr/benchmark/` |
| Tables 6–7 | `train_step1.py`, `train_step1_v2.py` | `results/ocr/step1*/predictions_test.json` |
| Table 8 | `ablation_eval.py` | `results/ablation/` |
| Table 9, §5.6.2 | `char_confusion_rates.py`, `quantify_error_types.py` | `results/error_analysis/` |

---

## Citation

Citation details will be added once the paper is published.

## License

Annotations and code are released under CC BY 4.0.
