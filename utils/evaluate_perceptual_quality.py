#!/usr/bin/env python3
"""
Evaluate perceptual quality of generated images using LAION aesthetic predictor.

This script:
1. Loads CLIP model to extract image embeddings
2. Loads LAION aesthetic predictor (linear layer on top of CLIP)
3. Computes aesthetic scores for all generated images in sub-action folders
4. Reports statistics grouped by step_id and sub-action, plus overall average

Usage:
    python utils/evaluate_perceptual_quality.py <results_dir>
    
Example:
    python utils/evaluate_perceptual_quality.py results/verifier_wo_reasoning_new_claim_gen_test_...
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from PIL import Image
try:
    import open_clip_torch as open_clip
except ImportError:
    import open_clip
from torchvision import transforms
from tqdm import tqdm


def get_aesthetic_model(clip_model="vit_l_14", device="cuda"):
    """Load the LAION aesthetic model weights from local aesthetic-predictor directory."""
    # Look for model weights in aesthetic-predictor/ directory
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent  # Go up from utils/ to workspace root
    
    # Try different possible locations
    possible_paths = [
        workspace_root / "aesthetic-predictor" / f"sa_0_4_{clip_model}_linear.pth",
        script_dir / ".." / "aesthetic-predictor" / f"sa_0_4_{clip_model}_linear.pth",
        Path(__file__).parent.parent / "aesthetic-predictor" / f"sa_0_4_{clip_model}_linear.pth",
    ]
    
    path_to_model = None
    for path in possible_paths:
        path = path.resolve()
        if path.exists():
            path_to_model = path
            break
    
    # If not found, try downloading to aesthetic-predictor/
    if path_to_model is None:
        aesthetic_dir = workspace_root / "aesthetic-predictor"
        aesthetic_dir.mkdir(exist_ok=True)
        path_to_model = aesthetic_dir / f"sa_0_4_{clip_model}_linear.pth"
        
        if not path_to_model.exists():
            print(f"Model weights not found. Please download sa_0_4_{clip_model}_linear.pth")
            print(f"  Expected location: {path_to_model}")
            print(f"  Download from: https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true")
            raise FileNotFoundError(f"Model weights not found at {path_to_model}")
    
    print(f"Loading aesthetic model from: {path_to_model}")
    
    # Create the linear layer based on CLIP model
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    elif clip_model == "vit_b_16":
        m = nn.Linear(512, 1)
    else:
        raise ValueError(f"Unsupported CLIP model: {clip_model}")
    
    # Load the trained weights
    s = torch.load(path_to_model, map_location=device)
    m.load_state_dict(s)
    m.eval()
    m = m.to(device)
    
    return m


def load_clip_model(clip_model="ViT-L/14", device="cuda"):
    """Load CLIP model for image embeddings."""
    print(f"Loading CLIP model: {clip_model}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained='openai'
    )
    model = model.to(device)
    model.eval()
    
    # Update preprocess to use torchvision transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    print("CLIP model loaded successfully.")
    return model, preprocess


def get_image_embedding(image_path, model, preprocess, device="cuda"):
    """Extract CLIP embedding for an image."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Explicitly delete the image tensor to free memory
        del image_tensor
        torch.cuda.empty_cache() if device == "cuda" else None
        
        return image_features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def find_all_generated_images(results_dir: str) -> List[Tuple[str, str, str, str]]:
    """
    Find all generated PNG images in the results directory.
    
    Returns:
        List of tuples: (q_id, step_id, sub_action, image_path)
    """
    results_path = Path(results_dir)
    image_files = []
    
    # Get all q_id directories (numeric directories)
    q_id_dirs = sorted([d for d in results_path.iterdir() 
                        if d.is_dir() and d.name.isdigit()])
    
    print(f"Found {len(q_id_dirs)} question directories")
    
    for q_id_dir in q_id_dirs:
        q_id = q_id_dir.name
        
        # Find all step_* directories
        step_dirs = sorted([d for d in q_id_dir.iterdir() 
                            if d.is_dir() and d.name.startswith('step_')])
        
        for step_dir in step_dirs:
            step_id = step_dir.name
            
            # Find sub-action directories (move-forward, turn-left, turn-right)
            sub_action_dirs = [d for d in step_dir.iterdir() 
                              if d.is_dir() and any(
                                  sa in d.name for sa in 
                                  ['move-forward', 'turn-left', 'turn-right']
                              )]
            
            for sub_action_dir in sub_action_dirs:
                sub_action = sub_action_dir.name
                
                # Find all PNG files in this sub-action directory
                png_files = list(sub_action_dir.glob('*.png'))
                
                for png_file in png_files:
                    # Only include sample images (not helper_img, img_0, etc.)
                    if 'sample_' in png_file.name or any(
                        name in png_file.name for name in 
                        ['sample_0.25', 'sample_0.5', 'sample_0.75', 
                         'sample_9', 'sample_18', 'sample_27']
                    ):
                        image_files.append((q_id, step_id, sub_action, str(png_file)))
    
    print(f"Found {len(image_files)} generated images")
    return image_files


def evaluate_aesthetic_quality(image_files: List[Tuple[str, str, str, str]], 
                               model, preprocess, aesthetic_model, 
                               device="cuda", batch_size=32):
    """Compute aesthetic scores for all images."""
    scores = []
    grouped_scores = defaultdict(list)
    
    print("Computing aesthetic scores for all images...")
    
    for q_id, step_id, sub_action, image_path in tqdm(image_files):
        # Get image embedding
        embedding = get_image_embedding(image_path, model, preprocess, device)
        
        if embedding is not None:
            # Compute aesthetic score
            with torch.no_grad():
                aesthetic_score = aesthetic_model(embedding).item()
            
            scores.append(aesthetic_score)
            
            # Group by step_id and sub_action
            grouped_scores[(step_id, sub_action)].append(aesthetic_score)
            
            # Explicitly delete embedding to free memory
            del embedding
            if device == "cuda":
                torch.cuda.empty_cache()
    
    return scores, grouped_scores


def print_statistics(scores: List[float], grouped_scores: Dict):
    """Print detailed statistics on aesthetic scores."""
    print("\n" + "="*80)
    print("PERCEPTUAL QUALITY EVALUATION RESULTS")
    print("="*80)
    
    # Overall statistics
    overall_avg = np.mean(scores)
    overall_std = np.std(scores)
    overall_min = np.min(scores)
    overall_max = np.max(scores)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total images evaluated: {len(scores)}")
    print(f"  Average aesthetic score: {overall_avg:.4f}")
    print(f"  Std deviation: {overall_std:.4f}")
    print(f"  Min score: {overall_min:.4f}")
    print(f"  Max score: {overall_max:.4f}")
    
    # Statistics by sub-action
    print(f"\n{'='*80}")
    print("STATISTICS BY SUB-ACTION:")
    print(f"{'='*80}")
    
    sub_action_stats = defaultdict(list)
    for (step_id, sub_action), scores_list in grouped_scores.items():
        sub_action_stats[sub_action].extend(scores_list)
    
    for sub_action in sorted(sub_action_stats.keys()):
        scores_list = sub_action_stats[sub_action]
        print(f"\n  {sub_action}:")
        print(f"    Count: {len(scores_list)}")
        print(f"    Average: {np.mean(scores_list):.4f}")
        print(f"    Std: {np.std(scores_list):.4f}")
        print(f"    Min: {np.min(scores_list):.4f}")
        print(f"    Max: {np.max(scores_list):.4f}")
    
    # Statistics by step_id
    print(f"\n{'='*80}")
    print("STATISTICS BY STEP_ID:")
    print(f"{'='*80}")
    
    step_stats = defaultdict(list)
    for (step_id, sub_action), scores_list in grouped_scores.items():
        step_stats[step_id].extend(scores_list)
    
    for step_id in sorted(step_stats.keys()):
        scores_list = step_stats[step_id]
        print(f"\n  {step_id}:")
        print(f"    Count: {len(scores_list)}")
        print(f"    Average: {np.mean(scores_list):.4f}")
        print(f"    Std: {np.std(scores_list):.4f}")
    
    # Detailed breakdown by step_id and sub_action
    print(f"\n{'='*80}")
    print("DETAILED BREAKDOWN (BY STEP_ID AND SUB-ACTION):")
    print(f"{'='*80}")
    
    for (step_id, sub_action) in sorted(grouped_scores.keys()):
        scores_list = grouped_scores[(step_id, sub_action)]
        print(f"\n  {step_id} / {sub_action}:")
        print(f"    Count: {len(scores_list)}")
        print(f"    Average: {np.mean(scores_list):.4f}")
        print(f"    Std: {np.std(scores_list):.4f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perceptual quality of generated images using LAION aesthetic predictor"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to the results directory containing generated images"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-L/14",
        help="CLIP model variant to use (default: ViT-L/14)"
    )
    parser.add_argument(
        "--aesthetic_model",
        type=str,
        default="vit_l_14",
        help="Aesthetic model variant (default: vit_l_14, also supports vit_b_32, vit_b_16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (not used yet, processes one at a time)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save results as JSON"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    # Generate output filename based on results directory if not explicitly provided
    if args.output_json is None:
        results_name = Path(args.results_dir).name
        if not results_name:
            results_name = Path(args.results_dir).parent.name
        args.output_json = f"results/aesthetic_scores_{results_name}.json"
    
    # Load models
    print("Loading models...")
    clip_model, preprocess = load_clip_model(args.clip_model, args.device)
    aesthetic_model = get_aesthetic_model(args.aesthetic_model, args.device)
    
    # Find all generated images
    print(f"\nScanning results directory: {args.results_dir}")
    image_files = find_all_generated_images(args.results_dir)
    
    if len(image_files) == 0:
        print("No generated images found!")
        sys.exit(1)
    
    # Evaluate aesthetic quality
    scores, grouped_scores = evaluate_aesthetic_quality(
        image_files, clip_model, preprocess, aesthetic_model, args.device, args.batch_size
    )
    
    # Print statistics
    print_statistics(scores, grouped_scores)
    
    # Save results if requested
    if args.output_json:
        results = {
            "overall": {
                "total_images": len(scores),
                "average": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            },
            "by_sub_action": {},
            "by_step_id": {},
            "detailed": {}
        }
        
        # By sub-action
        sub_action_stats = defaultdict(list)
        for (step_id, sub_action), scores_list in grouped_scores.items():
            sub_action_stats[sub_action].extend(scores_list)
        
        for sub_action in sorted(sub_action_stats.keys()):
            scores_list = sub_action_stats[sub_action]
            results["by_sub_action"][sub_action] = {
                "count": len(scores_list),
                "average": float(np.mean(scores_list)),
                "std": float(np.std(scores_list)),
                "min": float(np.min(scores_list)),
                "max": float(np.max(scores_list))
            }
        
        # By step_id
        step_stats = defaultdict(list)
        for (step_id, sub_action), scores_list in grouped_scores.items():
            step_stats[step_id].extend(scores_list)
        
        for step_id in sorted(step_stats.keys()):
            scores_list = step_stats[step_id]
            results["by_step_id"][step_id] = {
                "count": len(scores_list),
                "average": float(np.mean(scores_list)),
                "std": float(np.std(scores_list)),
                "min": float(np.min(scores_list)),
                "max": float(np.max(scores_list))
            }
        
        # Detailed
        for (step_id, sub_action) in sorted(grouped_scores.keys()):
            scores_list = grouped_scores[(step_id, sub_action)]
            key = f"{step_id}/{sub_action}"
            results["detailed"][key] = {
                "count": len(scores_list),
                "average": float(np.mean(scores_list)),
                "std": float(np.std(scores_list)),
                "min": float(np.min(scores_list)),
                "max": float(np.max(scores_list))
            }
        
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()

