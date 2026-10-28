"""
=============================================================================
CASTLE-Pegon -- Phase 02: Crop Line Images dari Bbox
=============================================================================
Input : data/processed/pegon_raw.json
Output: data/processed/crops/{KitabXX}/{record_id}.png
        data/processed/pegon_raw.json (updated dengan cropped_image_path)

Run: python src/phase02_crop.py
=============================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PEGON_RAW_JSON, PROCESSED_DIR, LOGS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

CROPS_DIR   = os.path.join(PROCESSED_DIR, "crops")
PADDING_PX  = 5       # padding di setiap sisi crop
OUTPUT_FMT  = "PNG"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.FileHandler(
            os.path.join(LOGS_DIR, "phase02_crop.log"), encoding="utf-8"
        ),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CROP FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def crop_line(image_path: str, bbox: dict, padding: int = PADDING_PX):
    """
    Crop satu baris dari halaman penuh menggunakan bbox persentase.
    Returns PIL Image atau None jika gagal.
    """
    if not os.path.isfile(image_path):
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        W, H = img.size

        x1 = int(bbox['x']     / 100.0 * W)
        y1 = int(bbox['y']     / 100.0 * H)
        x2 = int((bbox['x'] + bbox['width'])  / 100.0 * W)
        y2 = int((bbox['y'] + bbox['height']) / 100.0 * H)

        # Tambah padding, clamp ke batas image
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(W, x2 + padding)
        y2 = min(H, y2 + padding)

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return None

        return img.crop((x1, y1, x2, y2))

    except Exception as e:
        log.error(f"Crop error {image_path}: {e}")
        return None


def get_crop_path(record: dict, crops_dir: str) -> str:
    """Generate output path untuk crop."""
    kitab_dir = os.path.join(crops_dir, record['kitab_id'])
    os.makedirs(kitab_dir, exist_ok=True)
    return os.path.join(kitab_dir, f"{record['record_id']}.png")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CROPS_DIR, exist_ok=True)

    log.info("=" * 55)
    log.info("CASTLE-Pegon -- Phase 02: Crop Line Images")
    log.info("=" * 55)

    # Load records
    with open(PEGON_RAW_JSON, 'r', encoding='utf-8') as f:
        records = json.load(f)

    log.info(f"  Total records: {len(records)}")
    log.info(f"  Crops dir    : {CROPS_DIR}")

    stats = {
        'success' : 0,
        'skipped' : 0,  # sudah ada
        'no_image': 0,
        'failed'  : 0,
    }
    failures = []

    for i, rec in enumerate(records):
        record_id  = rec['record_id']
        image_path = rec.get('image_path', '')
        bbox       = rec.get('bbox', {})

        crop_path  = get_crop_path(rec, CROPS_DIR)

        # Update record dengan crop path
        rec['cropped_image_path'] = crop_path

        # Skip jika sudah ada (resume-friendly)
        if os.path.isfile(crop_path):
            stats['skipped'] += 1
            continue

        # Skip jika tidak ada image
        if not image_path or not os.path.isfile(image_path):
            stats['no_image'] += 1
            rec['cropped_image_path'] = ""
            failures.append({
                'record_id': record_id,
                'reason'   : 'no_source_image',
                'path'     : image_path,
            })
            continue

        # Crop
        cropped = crop_line(image_path, bbox, PADDING_PX)

        if cropped is None:
            stats['failed'] += 1
            rec['cropped_image_path'] = ""
            failures.append({
                'record_id': record_id,
                'reason'   : 'crop_failed',
                'path'     : image_path,
            })
            continue

        # Save
        try:
            cropped.save(crop_path, format=OUTPUT_FMT)
            stats['success'] += 1
        except Exception as e:
            stats['failed'] += 1
            rec['cropped_image_path'] = ""
            failures.append({
                'record_id': record_id,
                'reason'   : f'save_error: {e}',
                'path'     : crop_path,
            })

        # Progress
        if (i + 1) % 200 == 0:
            log.info(f"  Progress: {i+1}/{len(records)} "
                     f"(ok={stats['success']}, "
                     f"skip={stats['skipped']}, "
                     f"fail={stats['failed']+stats['no_image']})")

    # Save updated JSON
    with open(PEGON_RAW_JSON, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Save failure log
    if failures:
        fail_path = os.path.join(LOGS_DIR, "phase02_failures.json")
        with open(fail_path, 'w', encoding='utf-8') as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)

    # Summary
    total     = len(records)
    ok        = stats['success'] + stats['skipped']
    log.info(f"\n{'='*55}")
    log.info("PHASE 02 COMPLETE")
    log.info(f"{'='*55}")
    log.info(f"  Total    : {total}")
    log.info(f"  Success  : {stats['success']} (new crops)")
    log.info(f"  Skipped  : {stats['skipped']} (already exist)")
    log.info(f"  No image : {stats['no_image']}")
    log.info(f"  Failed   : {stats['failed']}")
    log.info(f"  OK rate  : {ok/total*100:.1f}%")
    log.info(f"  Crops dir: {CROPS_DIR}")
    log.info(f"\n  NEXT: Phase 03 -- Filter non-Pegon records")
    log.info(f"        Run: python src/phase03_filter.py")
    log.info("=" * 55)


if __name__ == "__main__":
    main()