#!/usr/bin/env python3
"""
Script to parse results/svc_test_verifier_new_1_8_8_2/results.json file and create a dataframe
grouped by question_id and action_name with the required columns.

The JSON structure is:
{
    "verification_metrics": {
        "question_id": {
            "0": {
                "action_name": {
                    "sub_action_name": {
                        "helpfulness_score": value,
                        "exploration_score": value,
                        ...
                    }
                },
                "helpful_score_threshold": value,
                "exploration_score_threshold": value
            }
        }
    }
}
"""

import json
import pandas as pd
import argparse
from pathlib import Path


def parse_verifier_results(json_file_path, output_csv_path=None):
    """
    Parse the verifier results JSON file and create a dataframe with the required structure.
    
    Args:
        json_file_path (str): Path to the results.json file
        output_csv_path (str, optional): Path for output CSV file. If None, uses default name.
    
    Returns:
        pd.DataFrame: Parsed dataframe
    """
    
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract verification_metrics
    verification_metrics = data.get('verification_metrics', {})
    
    # List to store all rows
    rows = []
    
    # Iterate through question_ids
    for question_id, question_data in verification_metrics.items():
        # Get the "0" level data (second level key)
        level_0_data = question_data.get('0', {})
        
        # Extract thresholds (shared across all actions for this question_id)
        helpful_score_threshold = level_0_data.get('helpful_score_threshold')
        exploration_score_threshold = level_0_data.get('exploration_score_threshold')
        
        # Iterate through action names (third level keys)
        for action_name, action_data in level_0_data.items():
            # Skip the threshold keys
            if action_name in ['helpful_score_threshold', 'exploration_score_threshold']:
                continue
            
            # Iterate through sub-actions (fourth level keys)
            for sub_action_name, sub_action_data in action_data.items():
                # Extract scores
                helpfulness_score = sub_action_data.get('helpfulness_score')
                exploration_score = sub_action_data.get('exploration_score')
                
                # Create row
                row = {
                    'question_id': question_id,
                    'action_name': action_name,
                    'sub_action_name': sub_action_name,
                    'helpful_score_threshold': helpful_score_threshold,
                    'exploration_score_threshold': exploration_score_threshold,
                    'helpfulness_score': helpfulness_score,
                    'exploration_score': exploration_score
                }
                
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by question_id and action_name for better organization
    df = df.sort_values(['question_id', 'action_name', 'sub_action_name'])
    
    # Set output path if not provided
    if output_csv_path is None:
        output_csv_path = 'verifier_results_parsed.csv'
    
    # Write to CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"DataFrame created with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique question_ids: {sorted(df['question_id'].unique())}")
    print(f"Unique action_names: {sorted(df['action_name'].unique())}")
    print(f"Results saved to: {output_csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Parse verifier results JSON and create CSV')
    parser.add_argument('--input', '-i', 
                       default='results/svc_test_verifier_new_1_8_8_2/results.json',
                       help='Path to input JSON file')
    parser.add_argument('--output', '-o',
                       default='verifier_results_parsed.csv',
                       help='Path to output CSV file')
    
    args = parser.parse_args()
    
    # Convert to absolute paths if relative
    input_path = Path(args.input)
    if not input_path.is_absolute():
        # Assume relative to project root
        project_root = Path(__file__).parent.parent.parent
        input_path = project_root / input_path
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        # Assume relative to current working directory
        output_path = Path.cwd() / output_path
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    try:
        df = parse_verifier_results(str(input_path), str(output_path))
        print("\nFirst few rows of the dataframe:")
        print(df.head(10))
        return 0
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
