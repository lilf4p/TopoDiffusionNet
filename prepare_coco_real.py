"""
Prepare COCO val2017 real images for TopoDiffusionNet training.

This script:
1. Reads COCO instance annotations to count objects per image
2. Copies real RGB images (NOT masks) renamed to {count}_{original_name}.jpg
3. The topological constraint c = number of object instances (0-dim topology)
4. Optionally filters by count range to keep meaningful distribution

Dataset format expected by TopoDiffusionNet: c_xxx.ext
where c = number of connected components (0-dim topological constraint)
"""

import os
import json
import shutil
from collections import Counter
from PIL import Image
import argparse


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO real images for TDN")
    parser.add_argument("--coco_img_dir", type=str,
                        default="datasets/val2017",
                        help="Path to COCO val2017 images")
    parser.add_argument("--coco_ann_file", type=str,
                        default="datasets/annotations/instances_val2017.json",
                        help="Path to COCO instance annotations JSON")
    parser.add_argument("--output_dir", type=str,
                        default="datasets/coco_real",
                        help="Output directory for prepared dataset")
    parser.add_argument("--min_count", type=int, default=1,
                        help="Minimum object count to include (default: 1)")
    parser.add_argument("--max_count", type=int, default=7,
                        help="Maximum object count to include (default: 7)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resize images to this size (default: 256)")
    parser.add_argument("--min_area", type=float, default=1024,
                        help="Minimum annotation area to count as object (filters tiny/noisy annotations)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load COCO annotations
    print(f"Loading annotations from {args.coco_ann_file}...")
    with open(args.coco_ann_file, 'r') as f:
        coco_data = json.load(f)

    # Build image_id -> filename mapping
    id_to_filename = {}
    for img_info in coco_data['images']:
        id_to_filename[img_info['id']] = img_info['file_name']

    # Count object instances per image (filtering by area and ignoring crowd annotations)
    image_instance_count = Counter()
    for ann in coco_data['annotations']:
        if ann.get('iscrowd', 0):
            continue  # Skip crowd annotations
        if ann.get('area', 0) < args.min_area:
            continue  # Skip tiny annotations
        image_instance_count[ann['image_id']] += 1

    # Distribution of instance counts
    count_distribution = Counter(image_instance_count.values())
    print("\nObject count distribution (all images with annotations):")
    for count in sorted(count_distribution.keys()):
        print(f"  c={count}: {count_distribution[count]} images")

    # Filter to desired range
    print(f"\nFiltering to count range [{args.min_count}, {args.max_count}]...")
    selected_images = {
        img_id: count
        for img_id, count in image_instance_count.items()
        if args.min_count <= count <= args.max_count
    }

    # Distribution after filtering
    filtered_distribution = Counter(selected_images.values())
    print(f"Selected {len(selected_images)} images:")
    for count in sorted(filtered_distribution.keys()):
        print(f"  c={count}: {filtered_distribution[count]} images")

    # Copy and rename images
    print(f"\nCopying images to {args.output_dir}...")
    copied = 0
    for img_id, count in selected_images.items():
        src_filename = id_to_filename.get(img_id)
        if src_filename is None:
            continue
        src_path = os.path.join(args.coco_img_dir, src_filename)
        if not os.path.exists(src_path):
            print(f"  WARNING: {src_path} not found, skipping")
            continue

        # New filename: c_originalname.jpg
        base_name = os.path.splitext(src_filename)[0]
        dst_filename = f"{count}_{base_name}.jpg"
        dst_path = os.path.join(args.output_dir, dst_filename)

        # Resize to target size and save as RGB
        try:
            img = Image.open(src_path).convert("RGB")
            # Resize to square (center crop + resize like the dataset loader does)
            w, h = img.size
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            img = img.crop((left, top, left + min_side, top + min_side))
            img = img.resize((args.image_size, args.image_size), Image.BICUBIC)
            img.save(dst_path, quality=95)
            copied += 1
        except Exception as e:
            print(f"  ERROR processing {src_path}: {e}")

    print(f"\nDone! Copied and resized {copied} images to {args.output_dir}")
    print(f"Dataset ready for TopoDiffusionNet training.")
    print(f"\nSample filenames:")
    sample_files = sorted(os.listdir(args.output_dir))[:10]
    for f in sample_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
