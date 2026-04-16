# PegonKitab — Dataset and Code for Multimodal Post-OCR Correction of Pegon Manuscripts

[![Paper](https://img.shields.io/badge/Paper-Pattern%20Recognition-blue)](https://doi.org/PLACEHOLDER)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-orange)](https://drive.google.com/PLACEHOLDER)

This repository contains the dataset, code, and evaluation scripts for the paper:

> **When Arabic OCR Fails: Benchmarking and Multimodal Post-OCR Correction for Pegon Manuscripts**  
> Syauqie Muhammad Marier, Bayu Nadya Kusuma, Md Mehedi Hassan Dorjoy, Linan Zhu, Xiangjie Kong  


---

## Overview

**Pegon** is a modified Arabic script used in Javanese and Sundanese Islamic manuscripts. Several Pegon-specific characters differ from standard Arabic by only one or two dots, causing systematic OCR failures when using Arabic-trained models.

This work introduces:
- **PegonKitab** — 2,801 annotated handwritten line images from 5 manuscripts
- **A 16-model OCR benchmark** spanning traditional OCR, transformer HTR, vision-language models, and Arabic-specialized systems
- **CASTLE-Pegon** — a multimodal post-OCR correction model (ByT5 + ViT) achieving 18.6% CER reduction over GPT-4o

---

## Dataset

### PegonKitab Statistics

| Kitab | Language | Script style | Train | Val | Test | Total |
|-------|----------|-------------|-------|-----|------|-------|
| Kitab01 | Javanese | — | 208 | 38 | 39 | 285 |
| Kitab02 | Javanese | — | 462 | 84 | 84 | 630 |
| Kitab03 | Sundanese | — | 197 | 36 | 36 | 269 |
| Kitab04 | Sundanese | — | 650 | 118 | 118 | 886 |
| Kitab05 | Javanese | — | 419 | 76 | 229 | 724 |
| **Total** | | | **1,936** | **352** | **506** | **2,794** |

### Download

| Resource | Link | Size |
|----------|------|------|
| Cropped line images (CroppedLines/) | [Google Drive](https://drive.google.com/drive/folders/1rtQpl-DuXsU3idGxg_jB9u5GrkWTlq8s?usp=share_link) | ~XX MB |
| Annotation splits (JSON) | This repository (`data/splits/`) | ~XX MB |

### Data Format

Each JSON file in `data/splits/` contains a list of records:

```json
{
  "record_id": "Kitab01_001",
  "kitab_id": "Kitab01",
  "language": "javanese",
  "image_filename": "Kitab01-001.png",
  "ground_truth": "...",
  "ocr_gpt4o": "..."
}
```

| Field | Description |
|-------|-------------|
| `record_id` | Unique identifier |
| `kitab_id` | Source manuscript (Kitab01–05) |
| `language` | `javanese` or `sundanese` |
| `image_filename` | Filename in CroppedLines/ |
| `ground_truth` | Human-annotated transcription |
| `ocr_gpt4o` | GPT-4o zero-shot OCR output (used as correction input) |

---

## OCR Benchmark Results

Full results on PegonKitab test set (506 lines). CER and WER are macro-averaged. RC = recognition coverage (% of inputs producing non-empty output).

| Tier | Model | RC (%) | CER | WER |
|------|-------|--------|-----|-----|
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

---

## CASTLE-Pegon Model

### Architecture

CASTLE-Pegon fuses visual and textual modalities for post-OCR correction:
- **Text branch**: ByT5 encoder (byte-level, no tokenization needed for Arabic script)
- **Visual branch**: ViT encoder (patch-level image features from line image)
- **Fusion**: Learned gating — α·text_features + (1−α)·visual_features
- **Decoder**: ByT5 decoder with auto-regressive correction

### Results

| Model | CER | WER | Δ vs GPT-4o |
|-------|-----|-----|-------------|
| GPT-4o (best zero-shot) | 0.4243 | 0.9986 | — |
| ByT5 text-only | 0.4134 | 0.9834 | −2.6% |
| CASTLE-Pegon v1 (ByT5-base + ViT) | 0.3635 | — | −14.3% |
| **CASTLE-Pegon v2 (ByT5-ft + ViT)** | **0.3456** | — | **−18.6%** |


## Repository Structure

```
pegon-kitab/
├── data/
│   ├── splits/
│   │   ├── train.json          # 1,936 training records
│   │   ├── val.json            # 352 validation records
│   │   └── test.json           # 506 test records
│   └── annotations/            # Raw Label Studio exports
├── src/
│   ├── train_byt5.py           # ByT5 text-only training
│   ├── train_castle.py         # CASTLE-Pegon multimodal training
│   ├── evaluate.py             # CER/WER evaluation
│   └── dataset.py              # Dataset loader
├── results/                    # Benchmark result files
├── docs/                       # Additional documentation
├── examples/                   # Usage examples and notebooks
├── requirements.txt
└── README.md
```

---



---

## License

- **Dataset (PegonKitab)**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)
- **Code**: [MIT License](LICENSE-CODE)

---

## Contact

For questions about the dataset or code, please contact:  
**Xiangjie Kong** — xjkong@ieee.org  
Zhejiang Key Laboratory of Visual Information Intelligent Processing,  
Zhejiang University of Technology, Hangzhou, China
