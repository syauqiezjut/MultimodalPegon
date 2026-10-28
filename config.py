"""
Konfigurasi terpusat CASTLE-Pegon
Paths are set through environment variables (see below).
"""
import os

# ── Paths ──
# BASE_DIR defaults to the repository root. Override with environment variables:
#   PEGON_BASE_DIR  : project root (data/, results/, models/ live here)
#   PEGON_KITAB_DIR : folder with the original page images (ImgKitab01..05) and
#                     Label Studio exports ("Kitab 01"/, ..., Kitab05.json); only
#                     needed to rebuild the splits from scratch (phase01-04)
#   PEGON_ENV       : "local" (default), "autodl", or "zjut"; controls which OCR
#                     engines are enabled below
BASE_DIR = os.environ.get("PEGON_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
ENV = os.environ.get("PEGON_ENV", "local")
KITAB_LABEL_DIR = os.environ.get("PEGON_KITAB_DIR", os.path.join(BASE_DIR, "data", "kitab_source"))
KITAB_IMAGE_DIR = KITAB_LABEL_DIR

# ── Project directories ──
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
SPLITS_DIR     = os.path.join(DATA_DIR, "splits")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
MODELS_DIR     = os.path.join(BASE_DIR, "models")

# ── Key output files ──
PEGON_RAW_JSON    = os.path.join(PROCESSED_DIR, "pegon_raw.json")
CROPS_DIR         = os.path.join(PROCESSED_DIR, "crops")
SPLIT_TRAIN       = os.path.join(SPLITS_DIR, "train.json")
SPLIT_VAL         = os.path.join(SPLITS_DIR, "val.json")
SPLIT_TEST        = os.path.join(SPLITS_DIR, "test.json")
KG_PKL            = os.path.join(PROCESSED_DIR, "kg", "pegon_kg.pkl")

# ── Kitab source mapping ──
KITAB_SOURCES = {
    "Kitab01": {
        "label_files": [os.path.join(KITAB_LABEL_DIR, "Kitab 01", "Kitab 01 - 1.json")],
        "image_dir"  : os.path.join(KITAB_IMAGE_DIR, "ImgKitab01"),
        "language"   : "javanese",
    },
    "Kitab02": {
        "label_files": [os.path.join(KITAB_LABEL_DIR, "Kitab 02", "Kitab02-01.json")],
        "image_dir"  : os.path.join(KITAB_IMAGE_DIR, "ImgKitab02"),
        "language"   : "javanese",
    },
    "Kitab03": {
        "label_files": [os.path.join(KITAB_LABEL_DIR, "Kitab 03", "Kitab 3 - 01.json")],
        "image_dir"  : os.path.join(KITAB_IMAGE_DIR, "ImgKitab03"),
        "language"   : "sundanese",
    },
    "Kitab04": {
        "label_dir"  : os.path.join(KITAB_LABEL_DIR, "Kitab 04"),
        "image_dir"  : os.path.join(KITAB_IMAGE_DIR, "ImgKitab04"),
        "language"   : "javanese",
        "multi_file" : True,
    },
    "Kitab05": {
        "label_files": [os.path.join(KITAB_LABEL_DIR, "Kitab05.json")],
        "image_dir"  : os.path.join(KITAB_IMAGE_DIR, "ImgKitab05"),
        "language"   : "javanese",
    },
}

# ── Split ratio ──
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED  = 42

# ── OCR engines ──
# Set True/False sesuai yang tersedia di environment
OCR_ENGINES = {
    "tesseract" : True,
    "easyocr"   : True,
    "qwen"      : ENV in ["local", "autodl"],
    "gpt4v"     : ENV == "local",
    "qari"      : ENV == "autodl",
    "hatformer" : ENV == "autodl",
}

# ── Model paths ──
BYT5_PATH = os.path.join(MODELS_DIR, "byt5-base")
VIT_PATH  = os.path.join(MODELS_DIR, "vit-base-patch16-224")

# ── Training ──
BATCH_SIZE   = 4   if ENV == "local" else 16
GRAD_ACCUM   = 1   if ENV == "local" else 2
N_STAGE1     = 1   if ENV == "local" else 20
N_STAGE2     = 0   if ENV == "local" else 20
LR_STAGE1    = 3e-4
LR_STAGE2    = 5e-5

if __name__ == "__main__":
    print(f"Environment : {ENV}")
    print(f"Base dir    : {BASE_DIR}")
    print(f"Kitab label : {KITAB_LABEL_DIR}")
    print(f"OCR engines : {[k for k,v in OCR_ENGINES.items() if v]}")
