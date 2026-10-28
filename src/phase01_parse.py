"""
=============================================================================
CASTLE-Pegon — Phase 01: Parse Label Studio JSON -> pegon_raw.json
=============================================================================
Run: python src/phase01_parse.py
=============================================================================
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import KITAB_SOURCES, PEGON_RAW_JSON, PROCESSED_DIR, LOGS_DIR


def find_image_file(image_dir: str, filename: str) -> str:
    if not filename or not image_dir:
        return ""
    basename = os.path.basename(filename)

    direct = os.path.join(image_dir, basename)
    if os.path.isfile(direct):
        return direct

    stripped = re.sub(r'^[0-9a-f]+-', '', basename)
    if stripped != basename:
        stripped_path = os.path.join(image_dir, stripped)
        if os.path.isfile(stripped_path):
            return stripped_path

    if os.path.isdir(image_dir):
        for f in os.listdir(image_dir):
            if f.lower() == basename.lower():
                return os.path.join(image_dir, f)
            if stripped and f.lower() == stripped.lower():
                return os.path.join(image_dir, f)
    return ""


def extract_image_filename(data_field: dict) -> str:
    if not data_field:
        return ""
    for key in ['image', 'ocr', 'img', 'file']:
        val = data_field.get(key, "")
        if val:
            return str(val)
    for val in data_field.values():
        if isinstance(val, str) and any(
            val.lower().endswith(ext)
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        ):
            return val
    return ""


def parse_annotations(annotations: list) -> list:
    if not annotations:
        return []
    ann = None
    for a in annotations:
        if a.get('result'):
            ann = a
            break
    if not ann:
        return []

    results = ann.get('result', [])
    by_id = defaultdict(dict)

    for item in results:
        item_id   = item.get('id', '')
        item_type = item.get('type', '')
        value     = item.get('value', {})

        if item_type == 'rectanglelabels':
            by_id[item_id]['bbox'] = {
                'x'       : value.get('x', 0),
                'y'       : value.get('y', 0),
                'width'   : value.get('width', 0),
                'height'  : value.get('height', 0),
                'rotation': value.get('rotation', 0),
            }
            by_id[item_id]['original_width']  = item.get('original_width', 0)
            by_id[item_id]['original_height'] = item.get('original_height', 0)

        elif item_type == 'textarea':
            texts = value.get('text', [])
            text  = texts[0].strip() if texts else ''
            by_id[item_id]['ground_truth'] = text

    lines = []
    for line_id, data in by_id.items():
        if 'bbox' not in data or 'ground_truth' not in data:
            continue
        lines.append({
            'line_id'        : line_id,
            'bbox'           : data['bbox'],
            'ground_truth'   : data['ground_truth'],
            'original_width' : data.get('original_width', 0),
            'original_height': data.get('original_height', 0),
        })

    lines.sort(key=lambda x: x['bbox']['y'])
    return lines


def load_json_files(file_paths: list) -> list:
    all_pages = []
    for path in file_paths:
        if not os.path.isfile(path):
            print(f"    WARNING: File tidak ditemukan: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            all_pages.extend(data)
        elif isinstance(data, dict):
            all_pages.append(data)
    return all_pages


def load_kitab_multifile(label_dir: str) -> list:
    """Load semua JSON dari folder (Kitab04 — per halaman)."""
    all_pages = []
    if not os.path.isdir(label_dir):
        print(f"    WARNING: Folder tidak ditemukan: {label_dir}")
        return []
    json_files = sorted(f for f in os.listdir(label_dir) if f.endswith('.json'))
    for fname in json_files:
        with open(os.path.join(label_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ekstrak page number dari nama file: "Kitab 04 #12.json" → 12
        page_num = None
        match = re.search(r'#(\d+)', fname)
        if match:
            page_num = int(match.group(1))

        if isinstance(data, list):
            for page in data:
                if page_num is not None:
                    page['inner_id'] = page_num
                all_pages.append(page)
        elif isinstance(data, dict):
            if page_num is not None:
                data['inner_id'] = page_num
            all_pages.append(data)

    return all_pages


def parse_kitab(kitab_id: str, kitab_config: dict) -> list:
    image_dir = kitab_config['image_dir']
    language  = kitab_config.get('language', 'javanese')

    if kitab_config.get('multi_file'):
        pages = load_kitab_multifile(kitab_config['label_dir'])
        n_files = len([f for f in os.listdir(kitab_config['label_dir'])
                       if f.endswith('.json')])
        print(f"  {kitab_id}: {n_files} JSON file(s) | lang={language}")
    else:
        file_paths = kitab_config['label_files']
        pages = load_json_files(file_paths)
        print(f"  {kitab_id}: {len(file_paths)} JSON file(s) | lang={language}")

    records = []

    for page in pages:
        inner_id    = page.get('inner_id', 0)
        data_field  = page.get('data', {})
        annotations = page.get('annotations', [])

        img_filename = extract_image_filename(data_field)
        img_path     = find_image_file(image_dir, img_filename)

        lines = parse_annotations(annotations)

        for line_order, line in enumerate(lines):
            gt = line['ground_truth']
            if not gt:
                continue

            record_id = f"{kitab_id}_p{inner_id:02d}_{line['line_id'][:10]}"

            records.append({
                'record_id'      : record_id,
                'kitab_id'       : kitab_id,
                'page_number'    : inner_id,
                'image_filename' : os.path.basename(img_path) if img_path else "",
                'image_path'     : img_path,
                'image_exists'   : os.path.isfile(img_path) if img_path else False,
                'line_id'        : line['line_id'],
                'line_order'     : line_order,
                'bbox'           : line['bbox'],
                'original_width' : line['original_width'],
                'original_height': line['original_height'],
                'ground_truth'   : gt,
                'language'       : language,
                'ocr_tesseract'  : None,
                'ocr_easyocr'    : None,
                'ocr_qwen'       : None,
                'ocr_gpt4v'      : None,
                'ocr_qari'       : None,
                'ocr_hatformer'  : None,
            })

    n_img = sum(1 for r in records if r['image_exists'])
    print(f"    -> {len(records)} line records | image found: {n_img}/{len(records)}")
    return records


def print_stats(records: list):
    print(f"\n{'='*55}")
    print("DATASET STATISTICS")
    print(f"{'='*55}")
    print(f"  Total records    : {len(records)}")

    by_kitab = Counter(r['kitab_id'] for r in records)
    by_lang  = Counter(r['language']  for r in records)

    print(f"\n  Per kitab:")
    for k in sorted(by_kitab):
        n     = by_kitab[k]
        n_img = sum(1 for r in records if r['kitab_id'] == k and r['image_exists'])
        lang  = next(r['language'] for r in records if r['kitab_id'] == k)
        print(f"    {k}: {n:>4} lines ({lang}) | image: {n_img}/{n}")

    n_exists = sum(1 for r in records if r['image_exists'])
    n_empty  = sum(1 for r in records if not r['ground_truth'].strip())
    lengths  = [len(r['ground_truth']) for r in records if r['ground_truth']]

    print(f"\n  Language distribution:")
    for lang, n in by_lang.items():
        print(f"    {lang}: {n}")

    print(f"\n  Image available  : {n_exists}/{len(records)} "
          f"({n_exists/len(records)*100:.1f}%)" if records else "")
    print(f"  Empty GT         : {n_empty}")

    if lengths:
        print(f"  GT length (chars):")
        print(f"    Min  : {min(lengths)}")
        print(f"    Max  : {max(lengths)}")
        print(f"    Mean : {sum(lengths)/len(lengths):.1f}")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 55)
    print("CASTLE-Pegon -- Phase 01: Parse Label Studio JSON")
    print("=" * 55)

    all_records = []
    for kitab_id, kitab_config in KITAB_SOURCES.items():
        try:
            records = parse_kitab(kitab_id, kitab_config)
            all_records.extend(records)
        except Exception as e:
            print(f"  ERROR {kitab_id}: {e}")
            import traceback
            traceback.print_exc()

    if not all_records:
        print("ERROR: No records parsed. Check config.py paths.")
        return

    print_stats(all_records)

    print(f"\nSaving to {PEGON_RAW_JSON}...")
    with open(PEGON_RAW_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(PEGON_RAW_JSON) / (1024**2)
    print(f"  OK {len(all_records)} records saved ({size_mb:.1f}MB)")

    print(f"\n  Sample records:")
    for r in all_records[:3]:
        img_status = "OK" if r['image_exists'] else "MISSING"
        print(f"  [{r['kitab_id']}] p{r['page_number']:02d} | "
              f"GT: '{r['ground_truth'][:30]}' | img: {img_status}")

    print(f"\n  NEXT: Phase 02 -- Crop line images")
    print(f"        Run: python src/phase02_crop.py")
    print("=" * 55)


if __name__ == "__main__":
    main()