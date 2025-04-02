import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons, CheckButtons, Slider
import tkinter as tk
import tkinter.messagebox as messagebox
import random
from PIL import Image
from shapely.geometry import Polygon
import datetime

# Define default paths
BASE_DIR = r"c:\Users\Ahmad\Desktop\Ozyegin University\research\datasets\DFC2023C"
DEFAULT_ORIGINAL_COCO_PATH = os.path.join(BASE_DIR, "buildings_only_train.json")
DEFAULT_PROCESSED_COCO_PATH = os.path.join(BASE_DIR, "processed_annotations.json")
DEFAULT_RGB_FOLDER = os.path.join(BASE_DIR, "train", "rgb")

# Define colors for different annotation types
COLORS = {
    "original": "green",
    "valid_building": "blue",
    "zero_height_building": "red",
    "invalid_polygon": "orange"
}

# Terminal logger class that outputs to both console and file
class TerminalLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')
        self.log_file.write("# JSON Annotation Viewer Log\n\n")
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

class JSONAnnotationViewer:
    def __init__(self, original_coco_path, processed_coco_path, rgb_folder):
        """Initialize the JSON annotation viewer with original and processed annotations"""
        # Store start time for session duration tracking
        self.start_time = datetime.datetime.now()
        
        # Store paths
        self.original_coco_path = original_coco_path
        self.processed_coco_path = processed_coco_path
        self.rgb_folder = rgb_folder
        
        print(f"# Initializing JSON Annotation Viewer")
        print(f"## Session Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nConfiguration:")
        print(f"- Original COCO file: `{self.original_coco_path}`")
        print(f"- Processed COCO file: `{self.processed_coco_path}`")
        print(f"- RGB folder: `{self.rgb_folder}`")
        print(f"\n---\n")
        
        # Load COCO data
        self.load_coco_data()
        
        # Setup the figure and UI
        self.setup_interface()
    
    def load_coco_data(self):
        """Load both original and processed COCO annotations"""
        print("## Loading Data")
        print("Loading original COCO data...")
        with open(self.original_coco_path) as f:
            self.original_coco = json.load(f)
        
        total_original = len(self.original_coco.get('annotations', []))
        print(f"Loaded {total_original} original annotations")
        
        # Index original annotations by image_id for faster lookup
        self.original_annotations_by_image = {}
        for ann in self.original_coco.get('annotations', []):
            image_id = ann.get('image_id')
            if image_id not in self.original_annotations_by_image:
                self.original_annotations_by_image[image_id] = []
            self.original_annotations_by_image[image_id].append(ann)
        
        print("Loading processed COCO data...")
        with open(self.processed_coco_path) as f:
            self.processed_coco = json.load(f)
        
        total_processed = len(self.processed_coco.get('annotations', []))
        print(f"Loaded {total_processed} processed annotations")
        
        # Count total polygons/building footprints (may be different from annotations count)
        total_polygons = 0
        for ann in self.processed_coco.get('annotations', []):
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                total_polygons += len(ann['segmentation'])
            else:
                # Count annotations without polygon segmentation or with RLE as 1
                total_polygons += 1
        
        # Index processed annotations by image_id for faster lookup
        self.processed_annotations_by_image = {}
        for ann in self.processed_coco.get('annotations', []):
            image_id = ann.get('image_id')
            if image_id not in self.processed_annotations_by_image:
                self.processed_annotations_by_image[image_id] = []
            self.processed_annotations_by_image[image_id].append(ann)
        
        # Create a list of all images
        self.images = self.processed_coco.get('images', [])
        total_images = len(self.images)
        print(f"Found {total_images} images")
        
        # Create different views based on annotation types
        self.create_views()
        
        # Log detailed statistics about annotation types
        print("\n## Annotation Statistics")
        
        # Add key counts summary before detailed tables
        print(f"\n### Key Counts")
        print(f"- Total Images: {total_images}")
        print(f"- Total Polygons/Building Footprints: {total_polygons}")
        print(f"- Total Building Annotations: {total_processed}")
        if total_polygons != total_processed:
            print(f"- Difference (polygons - annotations): {total_polygons - total_processed}")
            print(f"  (This difference occurs because some annotations contain multiple polygons)")
        
        # Create and print the image statistics table with better alignment
        image_stats_data = [
            ["Images in COCO File", total_images, "100.0%"]
        ]
        
        for view_name, image_ids in self.views.items():
            percentage = (len(image_ids) / total_images * 100) if total_images > 0 else 0
            image_stats_data.append([view_name, len(image_ids), f"{percentage:.1f}%"])
        
        self.print_markdown_table("Image Statistics", ["Category", "Count", "Percentage"], image_stats_data)
        
        # Count polygons by type
        valid_count = 0
        zero_height_count = 0
        invalid_count = 0
        
        for anns in self.processed_annotations_by_image.values():
            for ann in anns:
                label = ann.get("label", "")
                if label == "valid_building":
                    valid_count += 1
                elif label == "zero_height_building":
                    zero_height_count += 1
                elif label == "invalid_polygon":
                    invalid_count += 1
        
        # Create and print the polygon statistics table with better alignment
        polygon_stats_data = [
            ["Original Building Annotations", total_original, "100.0%"]
        ]
        
        processed_percentage = (total_processed / total_original * 100) if total_original > 0 else 0
        valid_percentage = (valid_count / total_processed * 100) if total_processed > 0 else 0
        zero_height_percentage = (zero_height_count / total_processed * 100) if total_processed > 0 else 0
        invalid_percentage = (invalid_count / total_processed * 100) if total_processed > 0 else 0
        
        polygon_stats_data.extend([
            ["Processed Building Annotations", total_processed, f"{processed_percentage:.1f}%"],
            ["Valid Building Annotations", valid_count, f"{valid_percentage:.1f}%"],
            ["Zero Height Building Annotations", zero_height_count, f"{zero_height_percentage:.1f}%"],
            ["Invalid Building Annotations", invalid_count, f"{invalid_percentage:.1f}%"]
        ])
        
        self.print_markdown_table("Building Annotation Statistics", ["Category", "Count", "Percentage"], polygon_stats_data)
        
        print("\n---\n")
    
    def print_markdown_table(self, title, headers, rows):
        """Print a well-formatted markdown table with consistent column widths"""
        # Calculate the maximum width for each column
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Add padding
        col_widths = [width + 2 for width in col_widths]
        
        # Print table title
        print(f"\n### {title}")
        
        # Print headers with proper padding
        header_line = "|"
        for i, header in enumerate(headers):
            header_line += f" {header:{col_widths[i]}} |"
        print(header_line)
        
        # Print separator line with proper widths
        separator_line = "|"
        for width in col_widths:
            separator_line += f" {'-' * width} |"
        print(separator_line)
        
        # Print data rows with proper padding
        for row in rows:
            row_line = "|"
            for i, cell in enumerate(row):
                row_line += f" {str(cell):{col_widths[i]}} |"
            print(row_line)
    
    def create_views(self):
        """Create different views based on annotation types"""
        # Create maps of image_ids to counts of different annotation types
        self.view_maps = {
            "all_images": set(),
            "with_valid_buildings": set(),
            "with_zero_height": set(),
            "with_invalid": set(),
            "with_all_three_types": set(),  # New view for images with all three types
            "with_mixed_types": set()  # New view for images with mixed annotation types
        }
        
        # First pass: collect image IDs for each annotation type
        annotation_types_by_image = {}
        for image_id, anns in self.processed_annotations_by_image.items():
            self.view_maps["all_images"].add(image_id)
            annotation_types_by_image[image_id] = set()
            
            # Check for each annotation type
            for ann in anns:
                label = ann.get("label", "")
                if label == "valid_building":
                    self.view_maps["with_valid_buildings"].add(image_id)
                    annotation_types_by_image[image_id].add("valid_building")
                elif label == "zero_height_building":
                    self.view_maps["with_zero_height"].add(image_id)
                    annotation_types_by_image[image_id].add("zero_height_building")
                elif label == "invalid_polygon":
                    self.view_maps["with_invalid"].add(image_id)
                    annotation_types_by_image[image_id].add("invalid_polygon")
        
        # Second pass: find images that have all three types or mixed types
        for image_id, types in annotation_types_by_image.items():
            if len(types) == 3:  # Image has all three types of annotations
                self.view_maps["with_all_three_types"].add(image_id)
                self.view_maps["with_mixed_types"].add(image_id)
            elif len(types) >= 2:  # Image has at least two different types
                self.view_maps["with_mixed_types"].add(image_id)
        
        # Convert to lists for random selection
        self.views = {
            "Images with Any Annotations": list(self.view_maps["all_images"]),
            "With Valid Buildings": list(self.view_maps["with_valid_buildings"]),
            "With Zero Height Buildings": list(self.view_maps["with_zero_height"]),
            "With Invalid Polygons": list(self.view_maps["with_invalid"]),
            "With All Three Types": list(self.view_maps["with_all_three_types"]),
            "With Mixed Types": list(self.view_maps["with_mixed_types"])
        }
        
        # Current view and index
        self.current_view_name = "Images with Any Annotations" 
        self.current_view = self.views[self.current_view_name]
        self.current_index = 0
        
        # Add explanation about the difference between total and annotated images
        images_without_annotations = len(self.images) - len(self.view_maps["all_images"])
        if images_without_annotations > 0:
            print(f"\n> Note: {images_without_annotations} image(s) in the COCO file have no annotations in the processed data.")
        
        # Remove redundant "View Breakdown" output as it's already included in "Image Statistics"
    
    def setup_interface(self):
        """Set up the matplotlib interface for viewing annotations"""
        # Create figure
        self.fig = plt.figure(figsize=(20, 10))
        
        # Create axes for the original and processed views
        self.ax_orig = plt.subplot2grid((1, 2), (0, 0))
        self.ax_proc = plt.subplot2grid((1, 2), (0, 1))
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2)
        
        # Set titles
        self.ax_orig.set_title("Original Polygons")
        self.ax_proc.set_title("Processed Annotations")
        
        # Add file path text box
        self.filepath_ax = plt.axes([0.20, 0.13, 0.6, 0.03])
        self.filepath_textbox = TextBox(
            self.filepath_ax, 'File Path:', 
            initial='No file selected',
            color='lightgoldenrodyellow'
        )
        # Disable editing but keep the text selectable
        self.filepath_textbox.disconnect_events()
        self.filepath_textbox.cursor.set_visible(False)
        
        # Add copy button for the file path
        self.copy_button_ax = plt.axes([0.81, 0.13, 0.07, 0.03])
        self.copy_button = Button(self.copy_button_ax, 'Copy Path')
        self.copy_button.on_clicked(self.copy_filepath)
        
        # Add sample navigation slider
        self.slider_ax = plt.axes([0.20, 0.09, 0.6, 0.03])
        max_samples = len(self.current_view)
        
        # Custom format function to display index in "current/total" format
        def format_slider_val(val):
            return f"{int(val)}/{max_samples}"
        
        self.sample_slider = Slider(
            self.slider_ax, 'Sample', 
            1, max(1, max_samples),  # 1-indexed
            valinit=1, valstep=1, valfmt='%s'
        )
        
        # Set the custom formatter
        self.sample_slider._format = format_slider_val
        self.sample_slider.valtext.set_text(format_slider_val(1))
        
        self.sample_slider.on_changed(self.update_sample)
        
        # Create navigation buttons
        self.prev_button_ax = plt.axes([0.40, 0.05, 0.07, 0.03])
        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.prev_button.on_clicked(self.prev_sample)
        
        self.next_button_ax = plt.axes([0.48, 0.05, 0.07, 0.03])
        self.next_button = Button(self.next_button_ax, 'Next')
        self.next_button.on_clicked(self.next_sample)
        
        self.random_button_ax = plt.axes([0.56, 0.05, 0.07, 0.03])
        self.random_button = Button(self.random_button_ax, 'Random')
        self.random_button.on_clicked(self.random_sample)
        
        # Create radio buttons to select view type
        self.radio_ax = plt.axes([0.01, 0.01, 0.15, 0.10])
        view_labels = list(self.views.keys())
        self.radio_buttons = RadioButtons(
            self.radio_ax, view_labels,
            active=view_labels.index(self.current_view_name)
        )
        self.radio_buttons.on_clicked(self.change_view)
        
        # Add a title for the radio buttons
        plt.figtext(0.01, 0.11, "Filter by annotation type:", fontsize=8,
                    bbox=dict(facecolor='lightgray', alpha=0.5, pad=2))
        
        # Create checkboxes for filtering annotation visibility
        self.check_ax = plt.axes([0.86, 0.01, 0.12, 0.10])
        self.check_buttons = CheckButtons(
            self.check_ax, ['Show valid', 'Show zero-height', 'Show invalid'],
            [True, True, True]
        )
        self.check_buttons.on_clicked(self.update_display)
        
        # Add a title for the checkboxes
        plt.figtext(0.86, 0.11, "Toggle visibility:", fontsize=8,
                    bbox=dict(facecolor='lightgray', alpha=0.5, pad=2))
        
        # Add legend
        self.add_legend()
        
        # Add synchronization note
        info_text = "Navigation synchronizes both views. Left panel shows original COCO polygons, right panel shows processed annotations."
        self.fig.text(0.5, 0.17, info_text, ha='center', fontsize=9, 
                     bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # Initialize the display with a random sample
        self.random_sample(None)
    
    def add_legend(self):
        """Add a legend explaining the colors"""
        legend_ax = plt.axes([0.41, 0.01, 0.20, 0.03])
        legend_ax.axis('off')
        
        # Create a horizontal legend with color squares
        x_positions = [0.05, 0.30, 0.55, 0.80]
        colors = [COLORS["original"], COLORS["valid_building"], 
                 COLORS["zero_height_building"], COLORS["invalid_polygon"]]
        labels = ["Original", "Valid", "Zero Height", "Invalid"]
        
        for i, (x, color, label) in enumerate(zip(x_positions, colors, labels)):
            legend_ax.add_patch(plt.Rectangle((x, 0.2), 0.05, 0.6, color=color))
            legend_ax.text(x + 0.06, 0.5, label, fontsize=8, va='center')
    
    def copy_filepath(self, event):
        """Copy the current filepath to clipboard"""
        filepath = self.filepath_textbox.text
        
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(filepath)
        messagebox.showinfo("Path Copied", f"File path copied to clipboard:\n{filepath}")
        root.update()
        root.destroy()
    
    def change_view(self, label):
        """Change the current view based on radio button selection"""
        self.current_view_name = label
        self.current_view = self.views[self.current_view_name]
        
        # Update slider range
        max_samples = len(self.current_view)
        self.sample_slider.valmax = max(1, max_samples)
        self.sample_slider.ax.set_xlim(1, max(1, max_samples))
        
        # Update the custom formatter
        self.sample_slider._format = lambda val: f"{int(val)}/{max_samples}"
        
        # Reset to first sample
        self.current_index = 0
        self.sample_slider.set_val(1)
        
        # Log the view change
        print(f"\n## View Changed to: {label}")
        print(f"Number of images in this view: {max_samples}")
        
        # Update the display
        self.update_display()
    
    def update_sample(self, val):
        """Update when the slider value changes"""
        self.current_index = int(val) - 1  # Convert from 1-indexed to 0-indexed
        self.update_display()
    
    def prev_sample(self, event):
        """Go to previous sample"""
        if len(self.current_view) == 0:
            return
        
        self.current_index = (self.current_index - 1) % len(self.current_view)
        self.sample_slider.set_val(self.current_index + 1)
        self.update_display()
    
    def next_sample(self, event):
        """Go to next sample"""
        if len(self.current_view) == 0:
            return
            
        self.current_index = (self.current_index + 1) % len(self.current_view)
        self.sample_slider.set_val(self.current_index + 1)
        self.update_display()
    
    def random_sample(self, event):
        """Go to a random sample"""
        if len(self.current_view) == 0:
            return
            
        self.current_index = random.randint(0, len(self.current_view) - 1)
        self.sample_slider.set_val(self.current_index + 1)
        self.update_display()
    
    def find_image_path(self, filename):
        """Find the full path to an image file, trying different extensions"""
        # First try with the original filename
        path = os.path.join(self.rgb_folder, filename)
        if os.path.exists(path):
            return path
        
        # Try with different extensions using the base name
        base_name = os.path.splitext(filename)[0]
        for ext in [".tif", ".tiff", ".TIF", ".TIFF", ".jpg", ".jpeg", ".png"]:
            path = os.path.join(self.rgb_folder, f"{base_name}{ext}")
            if os.path.exists(path):
                return path
        
        return None
    
    def update_display(self, *args):
        """Update the display with the current sample"""
        # Clear both axes
        self.ax_orig.clear()
        self.ax_proc.clear()
        
        # Check if we have a current view with images
        if len(self.current_view) == 0:
            self.ax_orig.text(0.5, 0.5, "No images in current view", ha='center', va='center')
            self.ax_proc.text(0.5, 0.5, "No images in current view", ha='center', va='center')
            self.ax_orig.axis('off')
            self.ax_proc.axis('off')
            self.fig.canvas.draw_idle()
            return
        
        # Get the current image ID
        image_id = self.current_view[self.current_index]
        
        # Find the image info for this ID
        image_info = next((img for img in self.images if img['id'] == image_id), None)
        if image_info is None:
            self.ax_orig.text(0.5, 0.5, f"Image info not found for ID: {image_id}", ha='center', va='center')
            self.ax_proc.text(0.5, 0.5, f"Image info not found for ID: {image_id}", ha='center', va='center')
            self.ax_orig.axis('off')
            self.ax_proc.axis('off')
            self.fig.canvas.draw_idle()
            return
        
        # Get the filename and find its path
        filename = image_info['file_name']
        img_path = self.find_image_path(filename)
        
        if img_path is None:
            self.ax_orig.text(0.5, 0.5, f"Image file not found: {filename}", ha='center', va='center')
            self.ax_proc.text(0.5, 0.5, f"Image file not found: {filename}", ha='center', va='center')
            self.ax_orig.axis('off')
            self.ax_proc.axis('off')
            self.fig.canvas.draw_idle()
            return
        
        # Update the file path text box
        self.filepath_textbox.set_val(img_path)
        
        try:
            # Load the image
            img = np.array(Image.open(img_path))
            
            # Display the image in both views
            self.ax_orig.imshow(img)
            self.ax_proc.imshow(img)
            
            # Update the titles
            basename = os.path.basename(img_path)
            index_info = f"Sample {self.current_index + 1}/{len(self.current_view)}"
            self.ax_orig.set_title(f"Original Polygons - {basename} ({index_info})")
            self.ax_proc.set_title(f"Processed Annotations - {basename} ({index_info})")
            
            # Get original annotations
            original_anns = self.original_annotations_by_image.get(image_id, [])
            
            # Show stats for original
            stats_text = f"Original Annotations: {len(original_anns)}"  # Updated terminology
            self.ax_orig.text(10, 20, stats_text, fontsize=9,
                             bbox=dict(facecolor='white', alpha=0.7))
            
            # Draw original annotations
            for ann in original_anns:
                # Skip annotations without segmentation
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                # Get segmentation data
                segmentation = ann['segmentation']
                
                if isinstance(segmentation, dict):
                    # RLE format - display placeholder
                    center_x, center_y, width, height = ann.get('bbox', [0, 0, 50, 50])
                    self.ax_orig.text(center_x + width/2, center_y + height/2,
                                    "RLE format", color='white', fontsize=8,
                                    bbox=dict(facecolor=COLORS["original"], alpha=0.5, pad=0.5),
                                    ha='center', va='center')
                elif isinstance(segmentation, list):
                    # Draw each polygon
                    for seg in segmentation:
                        if not isinstance(seg, list) or len(seg) < 6:
                            continue
                        
                        try:
                            # Convert flat list to array of [x,y] points
                            points = np.array(seg).reshape(-1, 2)
                            
                            # Draw polygon
                            polygon = plt.Polygon(points, fill=False, 
                                                edgecolor=COLORS["original"], linewidth=2, alpha=0.8)
                            self.ax_orig.add_patch(polygon)
                            
                            # Show area in the center of the polygon
                            if 'area' in ann:
                                area = ann['area']
                                centroid_x = points[:, 0].mean()
                                centroid_y = points[:, 1].mean()
                                
                                self.ax_orig.text(
                                    centroid_x, centroid_y,
                                    f"Area: {area:.1f}",
                                    fontsize=7,
                                    color='white',
                                    bbox=dict(facecolor=COLORS["original"], alpha=0.5, pad=0.5),
                                    ha='center', va='center'
                                )
                        except Exception as e:
                            print(f"Error plotting original polygon: {e}")
            
            # Get processed annotations
            processed_anns = self.processed_annotations_by_image.get(image_id, [])
            
            # Get visibility settings
            show_valid, show_zero_height, show_invalid = self.check_buttons.get_status()
            
            # Count annotations by type
            annotation_counts = {
                "valid_building": 0,
                "zero_height_building": 0,
                "invalid_polygon": 0
            }
            
            # Count visible annotations
            visible_counts = {
                "valid_building": 0,
                "zero_height_building": 0,
                "invalid_polygon": 0
            }
            
            # Draw processed annotations
            for ann in processed_anns:
                # Get label and color
                label = ann.get("label", "invalid_polygon")
                
                # Count this annotation
                if label in annotation_counts:
                    annotation_counts[label] += 1
                
                # Check if we should display this type
                show_this = False
                
                if label == "valid_building" and show_valid:
                    show_this = True
                    visible_counts["valid_building"] += 1
                    color = COLORS["valid_building"]
                elif label == "zero_height_building" and show_zero_height:
                    show_this = True
                    visible_counts["zero_height_building"] += 1
                    color = COLORS["zero_height_building"]
                elif label == "invalid_polygon" and show_invalid:
                    show_this = True
                    visible_counts["invalid_polygon"] += 1
                    color = COLORS["invalid_polygon"]
                else:
                    continue
                
                # Skip annotations without segmentation
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                # Get segmentation data
                segmentation = ann['segmentation']
                
                if isinstance(segmentation, dict):
                    # RLE format - display placeholder
                    center_x, center_y, width, height = ann.get('bbox', [0, 0, 50, 50])
                    self.ax_proc.text(center_x + width/2, center_y + height/2,
                                     "INVALID", color='white', fontsize=8,
                                     bbox=dict(facecolor=color, alpha=0.5, pad=0.5),
                                     ha='center', va='center')
                elif isinstance(segmentation, list):
                    if label == "invalid_polygon":
                        # Special handling for invalid polygons
                        for seg in segmentation:
                            if not isinstance(seg, list) or len(seg) < 6:
                                continue
                            
                            try:
                                points = np.array(seg).reshape(-1, 2)
                                # Just outline for invalid polygons
                                self.ax_proc.plot(points[:, 0], points[:, 1], '-o',
                                                color=color, linewidth=2, markersize=3, alpha=0.7)
                            except Exception as e:
                                print(f"Error plotting invalid polygon: {e}")
                    else:
                        # Regular polygon handling
                        for seg in segmentation:
                            if not isinstance(seg, list) or len(seg) < 6:
                                continue
                            
                            try:
                                points = np.array(seg).reshape(-1, 2)
                                polygon = plt.Polygon(points, fill=False,
                                                   edgecolor=color, linewidth=2, alpha=0.8)
                                self.ax_proc.add_patch(polygon)
                                
                                # Add height label for valid/zero-height buildings
                                if "average_height" in ann:
                                    height = ann["average_height"]
                                    centroid_x = points[:, 0].mean()
                                    centroid_y = points[:, 1].mean()
                                    
                                    label_text = f"{height:.1f}m" if label == "valid_building" else "0m"
                                    self.ax_proc.text(
                                        centroid_x, centroid_y,
                                        label_text,
                                        fontsize=7,
                                        color='white',
                                        bbox=dict(facecolor=color, alpha=0.5, pad=0.5),
                                        ha='center', va='center'
                                    )
                            except Exception as e:
                                print(f"Error plotting processed polygon: {e}")
            
            # Add annotation count info
            count_text = f"Valid={annotation_counts['valid_building']}, "
            count_text += f"Zero Height={annotation_counts['zero_height_building']}, "
            count_text += f"Invalid={annotation_counts['invalid_polygon']}"
            
            # Add mixed annotation type info
            unique_types = 0
            if annotation_counts['valid_building'] > 0:
                unique_types += 1
            if annotation_counts['zero_height_building'] > 0:
                unique_types += 1
            if annotation_counts['invalid_polygon'] > 0:
                unique_types += 1
            
            mixed_text = ""
            if unique_types >= 2:
                mixed_text = " (Mixed annotation types)"
            
            # Display at top left of processed view
            self.ax_proc.text(10, 20, count_text + mixed_text, fontsize=9,
                          bbox=dict(facecolor='white', alpha=0.7))
            
            # Log detailed statistics for the current image
            print(f"\n### Image Statistics for {os.path.basename(img_path)} (ID: {image_id})")
            print(f"- Original annotations: {len(original_anns)}")  # Updated terminology
            print(f"- Processed annotations: {len(processed_anns)}")
            print(f"- Valid buildings: {annotation_counts['valid_building']}")
            print(f"- Zero height buildings: {annotation_counts['zero_height_building']}")
            print(f"- Invalid buildings: {annotation_counts['invalid_polygon']}")  # Updated terminology
            
            if unique_types >= 2:
                print(f"- Has mixed annotation types: Yes ({unique_types} different types)")
            
            # Turn off axes for cleaner display
            self.ax_orig.axis('off')
            self.ax_proc.axis('off')
            
        except Exception as e:
            error_msg = f"Error loading/displaying image: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"Attempted path: {img_path}")
            
            self.ax_orig.text(0.5, 0.5, error_msg, ha='center', va='center')
            self.ax_proc.text(0.5, 0.5, error_msg, ha='center', va='center')
            
            self.ax_orig.text(0.5, 0.6, f"Path: {img_path}", ha='center', va='center', fontsize=8)
            self.ax_proc.text(0.5, 0.6, f"Path: {img_path}", ha='center', va='center', fontsize=8)
            
            self.ax_orig.axis('off')
            self.ax_proc.axis('off')
        
        # Update the figure
        self.fig.canvas.draw_idle()

def main():
    """Main function to launch the JSON annotation viewer"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="JSON Building Annotation Viewer - Show original and processed annotations side by side"
    )
    
    # Add arguments
    parser.add_argument(
        "--original-coco-path", 
        type=str, 
        default=DEFAULT_ORIGINAL_COCO_PATH,
        help="Path to the original COCO annotations file"
    )
    parser.add_argument(
        "--processed-coco-path", 
        type=str, 
        default=DEFAULT_PROCESSED_COCO_PATH,
        help="Path to the processed COCO annotations file"
    )
    parser.add_argument(
        "--rgb-folder", 
        type=str, 
        default=DEFAULT_RGB_FOLDER,
        help="Path to the image folder containing TIF/RGB images"
    )
    
    args = parser.parse_args()
    
    # Setup logging to both terminal and file
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        f"json_viewer_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    sys.stdout = TerminalLogger(log_path)
    
    # Print header
    print("# JSON Annotation Viewer")
    print("======================")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create and run the JSON annotation viewer
    viewer = JSONAnnotationViewer(
        args.original_coco_path,
        args.processed_coco_path,
        args.rgb_folder
    )
    
    plt.show()
    
    # Log session summary
    end_time = datetime.datetime.now()
    session_duration = end_time - viewer.start_time
    
    print("\n## Session Summary")
    print(f"- Session start: {viewer.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"- Session end: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"- Total duration: {session_duration}")
    print("\nViewer closed.")
    
    # Close and restore stdout
    if hasattr(sys.stdout, 'terminal'):
        orig_stdout = sys.stdout
        sys.stdout = sys.stdout.terminal
        orig_stdout.close()

if __name__ == "__main__":
    main()
