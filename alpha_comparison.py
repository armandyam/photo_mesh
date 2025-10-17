#!/usr/bin/env python3
"""
Alpha Comparison Script

This script runs the main photo mesh pipeline with different alpha values (0.0 to 1.0 in steps of 0.1)
and creates comparison grids showing all results side by side.
"""

import os
import cv2
import numpy as np
import subprocess
import argparse
from pathlib import Path

def run_pipeline_with_alpha(input_image, alpha):
    """Run the main pipeline with a specific alpha value."""
    cmd = ["python", "main.py", input_image, "--alpha", str(alpha)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running pipeline with alpha {alpha}: {result.stderr}")
        return None, None
    
    # Get the base name for output files
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    color_file = f"output/{base_name}_overlay_color_alpha{int(alpha*100)}.png"
    gray_file = f"output/{base_name}_overlay_gray_alpha{int(alpha*100)}.png"
    
    return color_file, gray_file

def create_comparison_grid(image_paths, output_path, grid_size=(5, 2)):
    """
    Create a comparison grid from a list of image paths.
    
    Args:
        image_paths: List of paths to images
        output_path: Path to save the comparison grid
        grid_size: (rows, cols) for the grid layout
    """
    if len(image_paths) == 0:
        print("No images to create grid from")
        return
    
    # Load all images
    images = []
    max_width, max_height = 0, 0
    
    for path in image_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                max_width = max(max_width, img.shape[1])
                max_height = max(max_height, img.shape[0])
            else:
                print(f"Could not load image: {path}")
        else:
            print(f"Image not found: {path}")
    
    if len(images) == 0:
        print("No valid images found")
        return
    
    # Resize all images to the same size
    resized_images = []
    for img in images:
        resized = cv2.resize(img, (max_width, max_height))
        resized_images.append(resized)
    
    # Create grid
    rows, cols = grid_size
    grid_height = max_height * rows
    grid_width = max_width * cols
    
    # Create the grid canvas
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place images in grid
    for i, img in enumerate(resized_images):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        y_start = row * max_height
        y_end = y_start + max_height
        x_start = col * max_width
        x_end = x_start + max_width
        
        grid[y_start:y_end, x_start:x_end] = img
    
    # Add alpha labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)  # White text
    thickness = 2
    
    for i in range(min(len(resized_images), rows * cols)):
        row = i // cols
        col = i % cols
        
        y_start = row * max_height
        x_start = col * max_width
        
        # Add alpha value label
        alpha_value = i * 0.1
        label = f"α={alpha_value:.1f}"
        
        # Position label at top-left of each image
        text_x = x_start + 10
        text_y = y_start + 30
        
        cv2.putText(grid, label, (text_x, text_y), font, font_scale, color, thickness)
    
    # Save the grid
    cv2.imwrite(output_path, grid)
    print(f"✅ Saved comparison grid: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create alpha comparison grid')
    parser.add_argument('input_image', help='Path to input image file')
    parser.add_argument('--output-dir', default='output', help='Output directory (default: output)')
    args = parser.parse_args()
    
    input_image = args.input_image
    output_dir = args.output_dir
    
    if not os.path.exists(input_image):
        print(f"❌ Error: Input image not found: {input_image}")
        return
    
    print(f"🚀 Creating alpha comparison for: {input_image}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images with different alpha values
    alpha_values = [i * 0.1 for i in range(11)]  # 0.0 to 1.0 in steps of 0.1
    color_paths = []
    gray_paths = []
    
    for alpha in alpha_values:
        print(f"🔄 Processing alpha = {alpha:.1f}")
        color_file, gray_file = run_pipeline_with_alpha(input_image, alpha)
        
        if color_file and gray_file:
            color_paths.append(color_file)
            gray_paths.append(gray_file)
        else:
            print(f"⚠️ Failed to generate images for alpha = {alpha:.1f}")
    
    # Create comparison grids
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    
    if color_paths:
        color_grid_path = os.path.join(output_dir, f"{base_name}_alpha_comparison_color.png")
        create_comparison_grid(color_paths, color_grid_path, grid_size=(2, 5))
    
    if gray_paths:
        gray_grid_path = os.path.join(output_dir, f"{base_name}_alpha_comparison_gray.png")
        create_comparison_grid(gray_paths, gray_grid_path, grid_size=(2, 5))
    
    print("✅ Alpha comparison complete!")
    print(f"📊 Generated {len(color_paths)} color images and {len(gray_paths)} gray images")
    print(f"🎨 Check the comparison grids in the output directory")

if __name__ == "__main__":
    main()
