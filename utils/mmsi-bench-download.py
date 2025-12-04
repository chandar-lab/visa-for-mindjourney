#!/usr/bin/env python3
"""
MMSI-Bench Dataset Download Script

This script downloads the MMSI-Bench dataset from HuggingFace, reads the parquet file,
decodes images from binary, and saves them as JPG files.

Based on instructions from: https://huggingface.co/datasets/RunsenXu/MMSI-Bench
"""

import os
import sys
import pandas as pd
from datasets import load_dataset
import argparse
from pathlib import Path
from PIL import Image
import io


def download_mmsi_bench(output_dir="./data/mmsi-bench/"):
    """
    Download MMSI-Bench dataset and save images as JPG files.
    
    Args:
        output_dir (str): Directory to save the dataset images
    """
    print("Starting MMSI-Bench dataset download...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create images subdirectory
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    try:
        # Load dataset from HuggingFace
        print("Loading MMSI-Bench dataset from HuggingFace...")
        mmsi_bench = load_dataset("RunsenXu/MMSI-Bench")
        print(f"Dataset loaded successfully: {mmsi_bench}")
        
        # Get the test split (which contains all the data)
        test_data = mmsi_bench['test']
        
        print(f"Processing {len(test_data)} records...")
        
        # Process each record
        for idx, row in enumerate(test_data):
            id_val = row['id']
            images = row['images']
            question_type = row['question_type']
            question = row['question']
            answer = row['answer']
            thought = row['thought']
            
            image_paths = []
            if images is not None:
                for n, img_data in enumerate(images):
                    image_path = images_dir / f"{id_val}_{n}.jpg"
                    
                    # Convert PIL Image to bytes if needed
                    if hasattr(img_data, 'save'):
                        # It's a PIL Image object
                        img_buffer = io.BytesIO()
                        img_data.save(img_buffer, format='JPEG')
                        img_bytes = img_buffer.getvalue()
                    else:
                        # It's already bytes
                        img_bytes = img_data
                    
                    with open(image_path, "wb") as f:
                        f.write(img_bytes)
                    image_paths.append(str(image_path))
            
            # Save metadata for each record
            metadata = {
                'id': id_val,
                'question_type': question_type,
                'question': question,
                'answer': answer,
                'thought': thought,
                'image_paths': image_paths,
                'num_images': len(image_paths)
            }
            
            # Save individual metadata file
            metadata_file = output_path / f"metadata_{id_val}.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(test_data)} records...")
        
        # Save complete dataset metadata
        print("Saving complete dataset metadata...")
        all_metadata = []
        for idx, row in enumerate(test_data):
            metadata = {
                'id': row['id'],
                'question_type': row['question_type'],
                'question': row['question'],
                'answer': row['answer'],
                'thought': row['thought'],
                'num_images': len(row['images']) if row['images'] else 0
            }
            all_metadata.append(metadata)
        
        # Save as CSV for easy viewing
        df = pd.DataFrame(all_metadata)
        csv_path = output_path / "mmsi_bench_metadata.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON for programmatic access
        json_path = output_path / "mmsi_bench_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print(f"Dataset download completed successfully!")
        print(f"Images saved to: {images_dir}")
        print(f"Metadata saved to: {output_path}")
        print(f"Total records processed: {len(test_data)}")
        print(f"Total images saved: {sum(len(row['images']) if row['images'] else 0 for row in test_data)}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download MMSI-Bench dataset")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data/mmsi-bench/",
        help="Directory to save the dataset (default: ./data/mmsi-bench/)"
    )
    
    args = parser.parse_args()
    
    download_mmsi_bench(args.output_dir)


if __name__ == "__main__":
    main()
