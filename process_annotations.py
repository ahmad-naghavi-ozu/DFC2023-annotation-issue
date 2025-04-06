import json
import os
import sys
import numpy as np
import time
import argparse
import traceback
from datetime import timedelta
from pathlib import Path
import cv2
import rasterio
from shapely.geometry import Polygon
from tqdm import tqdm

# Define relative paths - uses the current working directory as base
current_dir = os.path.abspath(os.path.dirname(__file__))

# Define JSON directory and ensure it exists
JSON_DIR = os.path.join(current_dir, "JSON")
os.makedirs(JSON_DIR, exist_ok=True)

DEFAULT_COCO_PATH = os.path.join(JSON_DIR, "buildings_only_train.json")
DEFAULT_OUTPUT_COCO_PATH = os.path.join(JSON_DIR, "processed_annotations.json")
DEFAULT_DSM_FOLDER = os.path.join(current_dir, "train", "dsm")

def decode_rle(rle, shape):
    """
    Decode RLE format to binary mask.
    
    Args:
        rle (dict): COCO RLE format with 'counts' and 'size' keys
        shape (tuple): Shape of the output mask (height, width)
        
    Returns:
        np.ndarray: Binary mask
    """
    if isinstance(rle['counts'], list):
        # Handle uncompressed RLE format
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        counts = rle['counts']
        pos = 0
        for i, count in enumerate(counts):
            val = i % 2  # 0, 1, 0, 1, ...
            mask[pos:pos+count] = val
            pos += count
        return mask.reshape(shape)
    else:
        # Handle compressed RLE format using pycocotools if possible
        try:
            from pycocotools import mask as coco_mask
            return coco_mask.decode(rle).astype(np.uint8)
        except ImportError:
            print("Warning: pycocotools not available, using fallback RLE decoder")
            # Fallback for compressed RLE - this is a simple implementation
            import zlib
            # Make sure numpy is available in this scope
            import numpy  # Import again to ensure it's available in this scope
            from itertools import groupby
            
            if isinstance(rle['counts'], bytes):
                # Decompress if the counts are compressed
                rle_counts = zlib.decompress(rle['counts']).decode('ascii')
            else:
                rle_counts = rle['counts']
                
            counts = [int(x) for x in rle_counts.split()]
            mask = numpy.zeros(shape[0] * shape[1], dtype=numpy.uint8)
            start = 0
            for i, count in enumerate(counts):
                val = i % 2
                end = start + count
                mask[start:end] = val
                start = end
            return mask.reshape(shape)

def process_annotations(coco_path, output_coco_path, dsm_folder, verbose=True):
    """
    Process COCO annotations to identify valid and zero-height building footprints.
    
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
    total_footprints = len(coco_data.get('annotations', []))
    invalid_count = 0
    zero_height_count = 0
    total_polygons = 0

    # Cache for DSM files to avoid reopening the same files
    dsm_cache = {}
    log_interval = 100  # Log progress every 100 annotations
    start_time = time.time()

    # Create image_id to image_info mapping for faster lookup
    image_info_map = {img["id"]: img for img in coco_data.get("images", [])}

    # Process each annotation with a progress bar
    if verbose:
        print("Processing annotations and computing heights...")
        print(f"This may take a while - processing {total_footprints} footprints...")
        print(f"Progress updates will be shown every {log_interval} annotations")

    for idx, ann in enumerate(tqdm(coco_data["annotations"], desc="Processing annotations", unit="annotation")):
        # Log progress periodically to show the script is still running
        if verbose and idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            annotations_per_sec = idx / elapsed if elapsed > 0 else 0
            est_remaining = (total_footprints - idx) / annotations_per_sec if annotations_per_sec > 0 else 0
            print(f"Processed {idx}/{total_footprints} annotations ({idx/total_footprints*100:.2f}%) - "
                  f"Speed: {annotations_per_sec:.2f} annotations/sec - "
                  f"Est. remaining: {timedelta(seconds=int(est_remaining))}")
        
        seg = ann.get("segmentation", [])
        is_valid = True
        is_rle_format = False
        
        # Handle different segmentation formats
        if isinstance(seg, dict) and 'counts' in seg:
            # RLE format - counts as 1 polygon
            is_rle_format = True
            total_polygons += 1
            # For RLE, check if decoding works without throwing an exception
            try:
                # Get the image info for shape
                image_id = ann["image_id"]
                image_info = image_info_map.get(image_id)
                if not image_info:
                    is_valid = False
                    continue
                
                height, width = image_info["height"], image_info["width"]
                # Try to decode the RLE - this will throw an exception if invalid
                mask = decode_rle(seg, (height, width))
                # If we get here, RLE decoding succeeded
            except Exception as e:
                if verbose:
                    print(f"Error decoding RLE: {e}")
                is_valid = False
        elif isinstance(seg, list) and len(seg) > 0:
            if isinstance(seg[0], list):
                # Polygon format - count each list as one polygon/part
                total_polygons += len(seg)
                # Check each polygon in the list
                for poly in seg:
                    if not poly or len(poly) < 6:  # less than 3 points
                        is_valid = False
                        break

                    try:
                        points = np.array(poly).reshape(-1, 2)
                        p = Polygon(points)
                        # We just check if a Shapely polygon can be created
                        # Don't use p.is_valid() as that's stricter than we need
                    except Exception as e:
                        if verbose:
                            print(f"Error creating polygon: {e}")
                        is_valid = False
                        break
            elif isinstance(seg[0], dict) and 'counts' in seg[0]:
                # List of RLE format - count each as one polygon/part
                is_rle_format = True
                total_polygons += len(seg)
                # Check if all RLEs can be decoded
                try:
                    # Get the image info for shape
                    image_id = ann["image_id"]
                    image_info = image_info_map.get(image_id)
                    if not image_info:
                        is_valid = False
                        continue
                    
                    height, width = image_info["height"], image_info["width"]
                    mask = np.zeros((height, width), dtype=np.uint8)
                    for rle in seg:
                        # Try to decode each RLE
                        rle_mask = decode_rle(rle, (height, width))
                        mask = np.logical_or(mask, rle_mask).astype(np.uint8)
                    # If we get here, all RLEs decoded successfully
                except Exception as e:
                    if verbose:
                        print(f"Error decoding RLE list: {e}")
                    is_valid = False
            else:
                # Unknown format
                is_valid = False
        else:
            # Empty or invalid segmentation format
            is_valid = False

        if not is_valid:
            ann["valid"] = False
            ann["color"] = "orange"
            ann["label"] = "invalid_building"
            invalid_count += 1
            continue

        # For valid building footprints, compute average height from the DSM
        image_id = ann["image_id"]
        image_info = image_info_map.get(image_id)
        if image_info is None:
            ann["valid"] = False
            ann["color"] = "orange"
            ann["label"] = "invalid_building"
            invalid_count += 1
            continue

        # Create the DSM path from the image filename
        img_filename = os.path.splitext(image_info['file_name'])[0]
        dsm_path = os.path.join(dsm_folder, f"{img_filename}.tif")
        
        if not os.path.exists(dsm_path):
            ann["valid"] = False
            ann["color"] = "orange"
            ann["label"] = "invalid_building"
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
            ann["label"] = "invalid_building"
            invalid_count += 1
            continue

        # Create a mask for this building footprint
        height, width = image_info['height'], image_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process based on segmentation format
        if is_rle_format:
            if isinstance(seg, dict):
                # Single RLE
                mask = decode_rle(seg, (height, width))
            else:
                # List of RLEs
                for rle in seg:
                    rle_mask = decode_rle(rle, (height, width))
                    mask = np.logical_or(mask, rle_mask).astype(np.uint8)
        else:
            # Regular polygon format
            for poly in seg:
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)

        # Calculate average height
        dsm_values = dsm[mask == 1]
        if dsm_values.size == 0:
            avg_height = 0
        else:
            avg_height = float(np.mean(dsm_values))
        
        ann["valid"] = True
        ann["average_height"] = avg_height
        
        # Color assignment: red for zero height, blue for valid non-zero height
        if np.isclose(avg_height, 0):
            ann["color"] = "red"
            ann["label"] = "zero_height_building"
            zero_height_count += 1
        else:
            ann["color"] = "blue"
            ann["label"] = "valid_building"

    # Display final timing information
    total_time = time.time() - start_time
    
    # Statistics
    stats = {
        "total_footprints": total_footprints,
        "total_polygons": total_polygons,
        "invalid_count": invalid_count,
        "zero_height_count": zero_height_count,
        "valid_buildings_count": total_footprints - invalid_count - zero_height_count,
        "processing_time_seconds": total_time,
        "annotations_per_second": total_footprints/total_time if total_time > 0 else 0
    }
    
    if verbose:
        print(f"\nProcessing completed in {timedelta(seconds=int(total_time))}!")
        print(f"Processing speed: {stats['annotations_per_second']:.2f} footprints per second")
        print(f"\nProcessing statistics:")
        print(f"Total building footprints: {stats['total_footprints']}")
        print(f"Total polygons in footprints: {stats['total_polygons']}")
        print(f"Difference (polygons - footprints): {stats['total_polygons'] - stats['total_footprints']}")
        print(f"Invalid building footprints: {stats['invalid_count']} ({(stats['invalid_count']/stats['total_footprints']*100):.2f}%)")
        print(f"Zero-height building footprints: {stats['zero_height_count']} ({(stats['zero_height_count']/(stats['total_footprints']-stats['invalid_count'])*100):.2f}% of valid footprints)")
        print(f"Valid building footprints with height: {stats['valid_buildings_count']} ({(stats['valid_buildings_count']/stats['total_footprints']*100):.2f}%)")

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
    parser = argparse.ArgumentParser(description="Process annotations to identify valid, zero-height, and invalid building footprints")
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
