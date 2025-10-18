#!/usr/bin/env python3
"""
Simulation Animation Script

This script generates an animation showing a person being "consumed by simulation"
by gradually increasing the mesh coverage from 0% to 100%.
"""

import os
import cv2
import numpy as np
import subprocess
import argparse
from pathlib import Path

def run_pipeline_with_cutoff(input_image, cutoff_position, alpha=0.6):
    """Run the main pipeline with a specific cutoff position."""
    cmd = [
        "python", "main.py", input_image, 
        "--alpha", str(alpha),
        "--cutoff-position", str(cutoff_position)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running pipeline with cutoff position {cutoff_position:.1f}: {result.stderr}")
        return None, None
    
    # Get the base name for output files
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    cutoff_suffix = f"_cutoff{int(cutoff_position*100):03d}"
    
    # Check if we have a mesh output or original image
    if cutoff_position <= 0:
        color_file = f"output/{base_name}_original.png"
        gray_file = f"output/{base_name}_original.png"
    else:
        color_file = f"output/{base_name}{cutoff_suffix}_overlay_color_alpha{int(alpha*100)}.png"
        gray_file = f"output/{base_name}{cutoff_suffix}_overlay_gray_alpha{int(alpha*100)}.png"
    
    return color_file, gray_file

def create_animation_frames(input_image, start_position=0.0, end_position=1.0, alpha=0.6, num_frames=21):
    """
    Create animation frames showing gradual consumption by simulation.
    
    Args:
        input_image: Path to input image
        start_position: Starting cutoff position (0.0-1.0, default: 0.0 for no meshing)
        end_position: Ending cutoff position (0.0-1.0, default: 1.0 for full meshing)
        alpha: Overlay transparency
        num_frames: Number of frames in animation (default: 21)
    """
    print(f"🎬 Creating simulation animation with {num_frames} frames")
    print(f"📍 Cutoff range: {start_position*100:.1f}% to {end_position*100:.1f}% from left")
    print(f"🎨 Alpha: {alpha}")
    
    # Generate frames
    color_frames = []
    gray_frames = []
    
    for i in range(num_frames):
        # Interpolate cutoff position from start to end
        cutoff_position = start_position + (end_position - start_position) * (i / (num_frames - 1))
        print(f"🔄 Frame {i+1}/{num_frames}: cutoff at {cutoff_position*100:.1f}% from left")
        
        color_file, gray_file = run_pipeline_with_cutoff(
            input_image, cutoff_position, alpha
        )
        
        if color_file and gray_file and os.path.exists(color_file):
            color_frames.append(color_file)
            gray_frames.append(gray_file)
        else:
            print(f"⚠️ Failed to generate frame {i+1}")
    
    return color_frames, gray_frames

def create_gif_from_frames(frame_paths, output_path, duration=200):
    """
    Create a GIF from a list of frame paths.
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Path to save the GIF
        duration: Duration per frame in milliseconds
    """
    if not frame_paths:
        print("No frames to create GIF from")
        return
    
    # Load all frames
    frames = []
    for path in frame_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                # Convert BGR to RGB for PIL
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img_rgb)
            else:
                print(f"Could not load frame: {path}")
        else:
            print(f"Frame not found: {path}")
    
    if not frames:
        print("No valid frames found")
        return
    
    # Create GIF using PIL
    try:
        from PIL import Image
        
        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"✅ Saved GIF: {output_path}")
        
    except ImportError:
        print("⚠️ PIL not available, skipping GIF creation")
        print("Install with: pip install Pillow")

def create_video_from_frames(frame_paths, output_path, fps=10):
    """
    Create a video from a list of frame paths.
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Path to save the video
        fps: Frames per second
    """
    if not frame_paths:
        print("No frames to create video from")
        return
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print("Could not load first frame")
        return
    
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Could not create video writer")
        return
    
    # Write frames
    for path in frame_paths:
        if os.path.exists(path):
            frame = cv2.imread(path)
            if frame is not None:
                video_writer.write(frame)
            else:
                print(f"Could not load frame: {path}")
        else:
            print(f"Frame not found: {path}")
    
    video_writer.release()
    print(f"✅ Saved video: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create simulation consumption animation')
    parser.add_argument('input_image', help='Path to input image file')
    parser.add_argument('--start-position', type=float, default=0.0, 
                       help='Starting cutoff position (0-1, default: 0.0)')
    parser.add_argument('--end-position', type=float, default=1.0, 
                       help='Ending cutoff position (0-1, default: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.6, 
                       help='Overlay alpha in [0,1] (default: 0.6)')
    parser.add_argument('--num-frames', type=int, default=21, 
                       help='Number of animation frames (default: 21)')
    parser.add_argument('--fps', type=float, default=10.0, 
                       help='Video frames per second (default: 10.0)')
    parser.add_argument('--output-dir', default='output', 
                       help='Output directory (default: output)')
    args = parser.parse_args()
    
    input_image = args.input_image
    output_dir = args.output_dir
    
    if not os.path.exists(input_image):
        print(f"❌ Error: Input image not found: {input_image}")
        return
    
    print(f"🚀 Creating simulation animation for: {input_image}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate animation frames
    color_frames, gray_frames = create_animation_frames(
        input_image, 
        args.start_position,
        args.end_position,
        args.alpha, 
        args.num_frames
    )
    
    # Create animations
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    
    if color_frames:
        # Create GIF
        gif_path = os.path.join(output_dir, f"{base_name}_simulation_consumption_color.gif")
        create_gif_from_frames(color_frames, gif_path, duration=int(1000/args.fps))
        
        # Create video
        video_path = os.path.join(output_dir, f"{base_name}_simulation_consumption_color.mp4")
        create_video_from_frames(color_frames, video_path, args.fps)
    
    if gray_frames:
        # Create GIF
        gif_path = os.path.join(output_dir, f"{base_name}_simulation_consumption_gray.gif")
        create_gif_from_frames(gray_frames, gif_path, duration=int(1000/args.fps))
        
        # Create video
        video_path = os.path.join(output_dir, f"{base_name}_simulation_consumption_gray.mp4")
        create_video_from_frames(gray_frames, video_path, args.fps)
    
    print("✅ Simulation animation complete!")
    print(f"🎬 Generated {len(color_frames)} color frames and {len(gray_frames)} gray frames")
    print(f"🎥 Check the GIF and MP4 files in the output directory")

if __name__ == "__main__":
    main()
