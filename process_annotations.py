import json
import os
import sys
import numpy as np
import time
import argparse
import traceback
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
import cv2
import rasterio
from shapely.geometry import Polygon

# Define default paths for better readability
BASE_DIR = r"c:\Users\Ahmad\Desktop\Ozyegin University\research\datasets\DFC2023C"
DEFAULT_COCO_PATH = os.path.join(BASE_DIR, "buildings_only_train.json")
DEFAULT_OUTPUT_COCO_PATH = os.path.join(BASE_DIR, "processed_annotations.json")
DEFAULT_DSM_FOLDER = os.path.join(BASE_DIR, "train", "dsm")

def process_annotations(coco_path, output_coco_path, dsm_folder, verbose=True):
    """
    Process COCO annotations to identify valid, zero-height, and invalid polygons.
    
    Args:
        coco_path: Path to the original COCO annotations file
        output_coco_path: Path to save the processed COCO annotations
        dsm_folder: Path to the DSM folder
        verbose: Whether to print progress information
    
    Returns:
        Dictionary containing statistics about the processing
    """
    if verbose:
        print(f"Loading original COCO annotations from {coco_path}...")
    
    with open(coco_path) as f:
        coco_data = json.load(f)
    
    if verbose:
        print(f"Loaded {len(coco_data.get('annotations', []))} annotations for {len(coco_data.get('images', []))} images")

    # Initialize counters for statistics
    total_polygons = len(coco_data.get('annotations', []))
    invalid_count = 0
    zero_height_count = 0

    # Cache for DSM files to avoid reopening the same files
    dsm_cache = {}
    log_interval = 100  # Log progress every 100 annotations
    start_time = time.time()

    # Process each annotation with a progress bar
    if verbose:
        print("Processing annotations and computing heights...")
        print(f"This may take a while - processing {total_polygons} annotations...")
        print(f"Progress updates will be shown every {log_interval} annotations")

    for idx, ann in enumerate(tqdm(coco_data["annotations"], desc="Processing annotations", unit="annotation")):
        # Log progress periodically to show the script is still running
        if verbose and idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            annotations_per_sec = idx / elapsed if elapsed > 0 else 0
            est_remaining = (total_polygons - idx) / annotations_per_sec if annotations_per_sec > 0 else 0
            print(f"Processed {idx}/{total_polygons} annotations ({idx/total_polygons*100:.2f}%) - "
                  f"Speed: {annotations_per_sec:.2f} annotations/sec - "
                  f"Est. remaining: {timedelta(seconds=int(est_remaining))}")
        
        seg = ann.get("segmentation", [])
        if not seg or not isinstance(seg, list):
            ann["valid"] = False
            ann["color"] = "orange"
            ann["label"] = "invalid_polygon"
            invalid_count += 1
            continue

        valid_polygon = True
        for poly in seg:
            if len(poly) < 6:  # less than 3 points
                valid_polygon = False
                break

            try:
                points = np.array(poly).reshape(-1, 2)
                p = Polygon(points)
            except Exception as e:
                if verbose:
                    print(f"Error creating polygon: {e}")
                valid_polygon = False
                break

        if not valid_polygon:
            ann["valid"] = False
            ann["color"] = "orange"
            ann["label"] = "invalid_polygon"
            invalid_count += 1
            continue

        # For valid polygons, compute average height from the corresponding DSM image
        image_id = ann["image_id"]
        image_info = next((img for img in coco_data["images"] if img["id"] == image_id), None)
        if image_info is None:
            ann["valid"] = False
            ann["color"] = "orange"
            ann["label"] = "invalid_polygon"
            invalid_count += 1
            continue

        dsm_path = os.path.join(dsm_folder, image_info["file_name"])
        if not os.path.exists(dsm_path):
            ann["valid"] = False
            ann["color"] = "orange"
            ann["label"] = "invalid_polygon"
            invalid_count += 1
            continue

        try:
            # Use cached DSM if available to improve performance
            if dsm_path in dsm_cache:
                dsm = dsm_cache[dsm_path]
            else:
                with rasterio.open(dsm_path) as src:
                    dsm = src.read(1)
                # Cache the DSM, but limit cache size to avoid memory issues
                if len(dsm_cache) < 100:  # Keep only the 100 most recent DSMs in cache
                    dsm_cache[dsm_path] = dsm
        except Exception as e:
            if verbose:
                print(f"Error reading DSM file {dsm_path}: {e}")
            ann["valid"] = False
            ann["color"] = "orange"
            ann["label"] = "invalid_polygon"
            invalid_count += 1
            continue

        # Create mask using all polygons
        mask = np.zeros(dsm.shape, dtype=np.uint8)
        for poly in seg:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

        dsm_values = dsm[mask == 1]
        if dsm_values.size == 0:
            avg_height = 0
        else:
            avg_height = float(np.mean(dsm_values))
        
        ann["valid"] = True
        ann["average_height"] = avg_height
        
        # Color assignment: red for zero height, blue for valid non-zero height
        if np.isclose(avg_height, 0):
            ann["color"] = "red"  # Ensure zero-height buildings are red
            ann["label"] = "zero_height_building"  # Correct label for zero-height buildings
            zero_height_count += 1
        else:
            ann["color"] = "blue"
            ann["label"] = "valid_building"  # Ensure valid buildings are labeled correctly

    # Display final timing information
    total_time = time.time() - start_time
    
    # Statistics
    stats = {
        "total_polygons": total_polygons,
        "invalid_count": invalid_count,
        "zero_height_count": zero_height_count,
        "valid_buildings_count": total_polygons - invalid_count - zero_height_count,
        "processing_time_seconds": total_time,
        "annotations_per_second": total_polygons/total_time if total_time > 0 else 0
    }
    
    if verbose:
        print(f"\nProcessing completed in {timedelta(seconds=int(total_time))}!")
        print(f"Processing speed: {stats['annotations_per_second']:.2f} annotations per second")
        print(f"\nProcessing statistics:")
        print(f"Total polygons: {stats['total_polygons']}")
        print(f"Invalid polygons: {stats['invalid_count']} ({(stats['invalid_count']/stats['total_polygons']*100):.2f}%)")
        print(f"Zero height buildings: {stats['zero_height_count']} ({(stats['zero_height_count']/(stats['total_polygons']-stats['invalid_count'])*100):.2f}% of valid polygons)")
        print(f"Valid buildings with height: {stats['valid_buildings_count']} ({(stats['valid_buildings_count']/stats['total_polygons']*100):.2f}%)")

    # Write updated annotations to new COCO file
    if verbose:
        print(f"\nWriting updated annotations to {output_coco_path}...")
    
    os.makedirs(os.path.dirname(output_coco_path), exist_ok=True)
    with open(output_coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    if verbose:
        print(f"Updated annotations saved to {output_coco_path}")
    
    return stats

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Process annotations to identify valid, zero-height, and invalid polygons")
    parser.add_argument("--coco-path", type=str, default=DEFAULT_COCO_PATH,
                       help="Path to the original COCO annotations file")
    parser.add_argument("--output-coco-path", type=str, default=DEFAULT_OUTPUT_COCO_PATH,
                       help="Path to save the processed COCO annotations")
    parser.add_argument("--dsm-folder", type=str, default=DEFAULT_DSM_FOLDER,
                       help="Path to the DSM folder")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    args = parser.parse_args()
    
    # Process annotations
    stats = process_annotations(
        args.coco_path, 
        args.output_coco_path, 
        args.dsm_folder,
        verbose=not args.quiet
    )
    
    return stats

if __name__ == "__main__":
    main()
