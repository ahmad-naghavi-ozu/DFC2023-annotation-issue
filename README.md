# Issues in DFC2023 Dataset Annotations

## Executive Summary

This document outlines notable issues discovered in the DFC2023 dataset that impact its reliability for benchmarking and evaluation. Based on comprehensive analysis of 1,773 samples:

1. **Widespread Zero-Height Building Annotations** (85.1% of images contain at least one)
2. **Mixed Annotation Quality** across the dataset (87.0% of images contain mixed annotation types)
3. **Rooftop vs. Building Base Annotations** (significant number of samples with incorrect annotation reference)

These issues affect a significant portion of the dataset and deserve attention from the DFC organizing committee.

## Note on Terminology

Throughout this document, we refer to different categories of building annotations:

- **Valid building annotations**: Building footprints with proper geometry and positive height values
- **Zero-height building annotations**: Building footprints that have been assigned a height value of zero
- **Invalid building annotations**: Building footprints with geometry that fails to meet standard GIS validity requirements
- **Mixed annotation types**: Images containing buildings from at least two different quality categories (valid, zero-height, or invalid)

## 1. Building Masks with Zero Height

### Overview
Analysis reveals that 7.2% of building footprints have been assigned zero height in the DSM annotations, severely compromising height estimation tasks. More significantly, these zero-height buildings appear in 85.1% of all images in the dataset.

### Detailed Analysis
- **Total building annotations**: 125,153
- **Zero height building polygons**: 9,060 (7.2% of all annotations)
- **Images containing zero-height buildings**: 1,509 (85.1% of all annotated images)

### Visual Observations
Visual examination shows that buildings with zero-height masks frequently correspond to smaller structures compared to their surroundings in the RGB images. This pattern suggests that DSM annotations may have been created prior to RGB imagery acquisition, leading to temporal misalignment in the dataset.

## 2. Building Footprint Polygon Quality

### Overview
Analysis of the building footprint polygons revealed a small number of geometrically invalid polygons that fail to meet standard GIS validity requirements.

### Validation Summary
- **Total building annotations**: 125,153
- **Invalid polygon annotations**: 483 (0.4% of all annotations)
- **Images containing invalid polygons**: 301 (17.0% of all annotated images)

### Criteria Used for Polygon Validation

The following criteria were used to identify invalid polygons in the dataset:

1. **Basic Polygon Structure**
   - Ensures segmentation data exists and is properly formatted
   - Checks if polygons have at least 3 points (`len(poly) >= 6` for flattened coordinates)

2. **Creation Validity**
   - Attempts to construct a valid Shapely Polygon object from the coordinates
   - If construction fails with an exception, the polygon is considered invalid

Note: While the code doesn't explicitly use `poly.is_valid()` or other Shapely validation methods, the basic structural checks capture most common polygon issues.

## 3. Rooftop vs. Building Base Annotations

### Overview
Analysis suggests that a significant number of building annotations in the dataset represent building rooftops rather than building bases (footprints). This is particularly problematic for building height estimation tasks, which fundamentally require the building base as the reference point for accurate measurements.

### Impact on Height Estimation
When a building polygon represents a rooftop instead of the building base:
- Height values become unreliable as they no longer represent the true building height
- Any height estimation model trained on this data will learn incorrect patterns
- Evaluation metrics using these annotations would yield misleading performance indicators

### Detection Challenges
Unlike other issues in the dataset, detecting rooftop vs. base annotation problems:
- Requires visual inspection of each sample
- Cannot be reliably automated through geometry or attribute analysis
- Demands comparison between the annotation and the actual building appearance in imagery

### Example Cases
The following samples have been visually identified as containing rooftop annotations instead of building base footprints:

- `SV_Berlin_52.5354_13.4826`
- `SV_Berlin_52.5303_13.4828`
- `SV_Rio_-22.9068_-43.1956`
- `SV_Rio_-22.9077_-43.1803`

These examples represent just a subset of the affected samples. More instances will be documented as they are identified through ongoing visual inspection.

## Additional Dataset Statistics

Based on our comprehensive analysis:

- **Total samples**: 1,773
- **Total polygons/building footprints**: 125,774
- **Total building annotations**: 125,153
- **Difference (polygons - annotations)**: 621
  (This difference occurs because some annotations contain multiple polygons)

### Sample Distribution:
- Images in COCO file: 1,773 (100.0%)
- Images with any annotations: 1,770 (99.8%)
- Images with valid buildings: 1,769 (99.8% of all images)
- Images with zero height buildings: 1,509 (85.1% of all images)
- Images with invalid polygons: 301 (17.0% of all images)
- Images with all three types: 266 (15.0% of all images)
- Images with mixed annotation types: 1,543 (87.0% of all images)
  *(Mixed annotation types refers to images containing buildings from at least 2 different quality categories - valid, zero-height, or invalid)*

### Building Annotation Distribution:
- Total building annotations: 125,153 (100.0%)
- Valid building annotations: 115,610 (92.4% of all annotations)
- Zero height building annotations: 9,060 (7.2% of all annotations)
- Invalid building annotations: 483 (0.4% of all annotations)

## Recommendations for Action

We respectfully request that the DFC23 organizing committee consider the following actions:

### Regarding Building Mask Corrections:
- Revisit the DSM annotations and update zero-height building footprints
- Ensure temporal consistency between DSM acquisition and RGB imagery
- Consider providing an updated version of the dataset with corrected height values

### Regarding Documentation:
- Share details about the DSM generation process to help users understand limitations
- Provide clarification on the temporal aspects of data collection
- Consider documenting known issues for the benefit of contest participants

## Conclusion

While the invalid polygon issue affects only a small percentage of the dataset (0.4% of annotations), the zero-height building issue is more pervasive, appearing in 85.1% of all images. These issues may impact the reliability of the DFC23 dataset for building height estimation tasks specifically.

We believe addressing these problems will enhance the value of this dataset as a benchmark for the remote sensing and geospatial analysis community.

Thank you for your attention to these matters.

## Working with the Dataset Tools

This repository includes tools to help analyze and visualize the DFC2023 dataset annotations. These tools can help researchers understand the dataset's characteristics and issues.

### Prerequisites

- The DFC2023 training data, which includes:
  - RGB imagery (in `train/rgb/`)
  - DSM files (in `train/dsm/`)
  - Building annotations (`buildings_only_train.json`)
- Python environment with required dependencies: numpy, matplotlib, rasterio, shapely, PIL, tqdm, cv2

> **Note:** The training images are not included in this GitHub repository due to storage space limitations. You will need to download them separately from the original DFC2023 competition source: [IEEE GRSS Data Fusion Contest 2023](https://ieee-dataport.org/competitions/2023-ieee-grss-data-fusion-contest-large-scale-fine-grained-building-classification)

### Processing Annotations

The `process_annotations.py` script analyzes the original COCO annotations and identifies valid buildings, zero-height buildings, and invalid polygons.

**Usage:**

```bash
python process_annotations.py [--coco-path COCO_PATH] [--output-coco-path OUTPUT_PATH] [--dsm-folder DSM_FOLDER] [--quiet]
```

**Key functionality:**
- Validates polygon geometry for each building annotation
- Calculates average height for each building using DSM data
- Classifies buildings into three categories: valid, zero-height, and invalid
- Generates an enhanced annotation file with added metadata
- Outputs statistics about the dataset

By default, the script reads from `buildings_only_train.json` and writes to `processed_annotations.json`, but these paths can be customized via command-line arguments.

> **Note:** For convenience, we have already included the processed annotations file `processed_annotations.json` in this repository, so you can skip the processing step and directly use the visualization tool.

### Visualizing Annotations

The `json_viewer.py` script provides a graphical interface for comparing original and processed annotations.

**Usage:**

```bash
python json_viewer.py [--original-coco-path ORIGINAL_PATH] [--processed-coco-path PROCESSED_PATH] [--rgb-folder RGB_FOLDER]
```

**Key functionality:**
- Side-by-side visualization of original and processed annotations
- Color-coding of different annotation types (blue for valid, red for zero-height, orange for invalid)
- Filtering options to view specific subsets of images (e.g., only those with zero-height buildings)
- Toggle visibility of different annotation types
- Navigation controls (previous, next, random)
- Sample count information and statistics
- Automatic logging of session activity to a markdown file (`json_viewer_log_YYYYMMDD_HHMMSS.md`) for further analysis

The viewer will help identify problematic annotations visually and assist in understanding the spatial distribution of annotation issues.

### Workflow

1. First, process the original annotations (optional, as processed annotations are already provided):
   ```bash
   python process_annotations.py
   ```

2. Then, view the processed annotations to analyze issues:
   ```bash
   python json_viewer.py
   ```

3. Use the viewer's filtering options to explore specific issues, such as viewing only images with mixed annotation types or zero-height buildings.

4. Review the automatically generated log file after your session for a record of the images you examined.

This workflow enables comprehensive analysis of the dataset annotation issues described in this document.
