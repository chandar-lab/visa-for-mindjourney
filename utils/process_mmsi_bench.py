#!/usr/bin/env python3
"""
MMSI-Bench Data Processing Script

This script processes the MMSI-Bench dataset and converts it to the same format
as the SAT dataset (data/test.json) for compatibility with existing data loading code.

Field mappings:
- "id" in mmsi => "database_idx" 
- "question_type" and "question" are the same but question should be without the Options
- "answer_choices" field needs to be parsed from the text after "Options:" in the question
- "answer" => "correct_answer" (but should be the full choice text, not just the letter)
- "image_paths" => "img_paths"
"""

import os
import sys
import json
import re
import random
import unicodedata
from pathlib import Path
import argparse


def normalize_unicode(text):
    """
    Normalize unicode characters in text.
    
    Args:
        text (str): Input text with potential unicode characters
        
    Returns:
        str: Normalized text with unicode characters replaced
    """
    if not text:
        return text
    
    # Replace common unicode characters with ASCII equivalents
    replacements = {
        '\u2013': '-',  # en dash
        '\u2014': '--', # em dash
        '\u2018': "'",  # left single quotation mark
        '\u2019': "'",  # right single quotation mark
        '\u201c': '"',  # left double quotation mark
        '\u201d': '"',  # right double quotation mark
        '\u2026': '...', # horizontal ellipsis
    }
    
    normalized = text
    for unicode_char, ascii_char in replacements.items():
        normalized = normalized.replace(unicode_char, ascii_char)
    
    return normalized


def clean_trailing_characters(text):
    """
    Remove trailing characters like semicolons, periods, etc. from text.
    
    Args:
        text (str): Input text that may have trailing characters
        
    Returns:
        str: Cleaned text with trailing characters removed
    """
    if not text:
        return text
    
    # List of trailing characters to remove
    trailing_chars = [';', '.', ',', ':', '!', '?', ' ', '\t', '\n', '\r']
    
    cleaned = text
    # Remove trailing characters from the end
    while cleaned and cleaned[-1] in trailing_chars:
        cleaned = cleaned[:-1]
    
    return cleaned


def parse_answer_choices(question_text):
    """
    Parse answer choices from the question text after "Options:".
    
    Args:
        question_text (str): The full question text containing options
        
    Returns:
        tuple: (clean_question, answer_choices)
    """
    if not question_text:
        return question_text, []
    
    # Split on "Options:" to separate question from choices
    if "Options:" in question_text:
        question_part, options_part = question_text.split("Options:", 1)
        clean_question = question_part.strip()
        
        # Parse the options part
        answer_choices = []
        
        # More robust pattern that handles commas within choices
        # Look for "A: text, B: text, C: text, D: text" pattern
        # This pattern captures everything between A: and the next letter: or end of string
        option_pattern = r'([A-D]):\s*([^A-D]*(?:[A-D][^A-D]*)*?)(?=\s*[A-D]:|$)'
        matches = re.findall(option_pattern, options_part)
        
        for letter, choice in matches:
            # Clean up the choice text - remove trailing commas and whitespace
            choice_text = choice.strip().rstrip(',').strip()
            if choice_text:  # Only add non-empty choices
                cleaned_choice = clean_trailing_characters(choice_text.lower())
                answer_choices.append(cleaned_choice)
        
        # If still no matches found, try a simpler approach
        if not answer_choices:
            # Split by letter patterns and clean up
            parts = re.split(r'\s*([A-D]):\s*', options_part)
            if len(parts) > 1:
                for i in range(1, len(parts), 2):  # Skip the first empty part
                    if i + 1 < len(parts):
                        letter = parts[i]
                        choice_text = parts[i + 1].strip().rstrip(',').strip()
                        if choice_text:
                            cleaned_choice = clean_trailing_characters(choice_text.lower())
                            answer_choices.append(cleaned_choice)
        
        return clean_question, answer_choices
    else:
        # Fallback if no "Options:" found
        return question_text, []


def map_question_type(mmsi_question_type):
    """
    Map MMSI-Bench question types to SAT dataset equivalents.
    
    Args:
        mmsi_question_type (str): The question type from MMSI-Bench
        
    Returns:
        str: Mapped question type for SAT format
    """
    if not mmsi_question_type:
        return "unknown"
    
    # Normalize unicode characters first
    normalized_type = normalize_unicode(mmsi_question_type)
    
    # Keep MMSI-Bench question types as they are, just normalize unicode and formatting
    type_mapping = {
        "Motion (Cam.)": "motion_(cam.)",
        "Positional Relationship (Cam.-Obj.)": "positional_relationship_(cam.-obj.)", 
        "Positional Relationship (Reg.-Reg.)": "positional_relationship_(reg.-reg.)",
        "Positional Relationship (Obj.-Obj.)": "positional_relationship_(obj.-obj.)",
        "Positional Relationship (Cam.-Cam.)": "positional_relationship_(cam.-cam.)",
        "Positional Relationship (Cam.-Reg.)": "positional_relationship_(cam.-reg.)",
        "Positional Relationship (Obj.-Reg.)": "positional_relationship_(obj.-reg.)",
        "Action Consequence": "action_consequence",
        "Goal Aim": "goal_aim",
        "Motion (Obj.)": "motion_(obj.)",
        "Attribute (Appr.)": "attribute_(appr.)",
        "Attribute (Meas.)": "attribute_(meas.)",
        "MSR": "msr"
    }
    
    return type_mapping.get(normalized_type, normalized_type.lower().replace(" ", "_"))


def get_correct_answer_text(answer_letter, answer_choices):
    """
    Get the full text of the correct answer based on the letter.
    
    Args:
        answer_letter (str): The letter answer (A, B, C, D)
        answer_choices (list): List of answer choice texts (without labels)
        
    Returns:
        str: The full text of the correct answer
    """
    if not answer_choices or len(answer_choices) == 0:
        # If no answer choices available, return the letter as lowercase and cleaned
        return clean_trailing_characters(answer_letter.lower()) if answer_letter else ""
    
    # Map letter to index
    letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    index = letter_to_index.get(answer_letter.upper(), -1)
    
    # Check if the index is valid for the available choices
    if 0 <= index < len(answer_choices):
        # Return the choice text (already cleaned in parse_answer_choices)
        return answer_choices[index]
    else:
        # Handle mismatched answer labels - use the first available choice as fallback
        if answer_choices:
            print(f"Warning: Answer letter '{answer_letter}' doesn't match available choices {len(answer_choices)}. Using first choice as fallback.")
            return answer_choices[0]
        else:
            # Fallback to letter, cleaned
            return clean_trailing_characters(answer_letter.lower())


def process_mmsi_bench_data(mmsi_data_dir, output_file, num_questions_per_type=None, seed=42):
    """
    Process MMSI-Bench data and convert to SAT format.
    
    Args:
        mmsi_data_dir (str): Path to the MMSI-Bench data directory
        output_file (str): Path to the output test.json file
        num_questions_per_type (int, optional): Number of questions to select per type
        seed (int): Random seed for deterministic shuffling
    """
    print("Processing MMSI-Bench data...")
    
    mmsi_path = Path(mmsi_data_dir)
    if not mmsi_path.exists():
        print(f"Error: MMSI-Bench data directory not found: {mmsi_data_dir}")
        sys.exit(1)
    
    # Load the complete metadata
    metadata_file = mmsi_path / "mmsi_bench_metadata.json"
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        sys.exit(1)
    
    with open(metadata_file, 'r') as f:
        mmsi_data = json.load(f)
    
    print(f"Loaded {len(mmsi_data)} records from MMSI-Bench metadata")
    
    # Pre-filter: Only process questions with at most 2 images
    filtered_data = [record for record in mmsi_data if record.get('num_images', 0) <= 2]
    print(f"Pre-filtered to {len(filtered_data)} records with at most 2 images (removed {len(mmsi_data) - len(filtered_data)} records with >2 images)")
    
    # Process each record and group by question type
    questions_by_type = {}
    
    skipped_records = []
    for record in filtered_data:
        try:
            # Validate required fields
            if not record.get('id') or not record.get('question'):
                print(f"Warning: Skipping record with missing required fields: {record.get('id', 'unknown')}")
                skipped_records.append(record.get('id', 'unknown'))
                continue
            
            # Parse question and answer choices
            clean_question, answer_choices = parse_answer_choices(record['question'])
            
            # Map question type (with unicode normalization)
            sat_question_type = map_question_type(record['question_type'])
            
            # Get the full text of the correct answer
            correct_answer_text = get_correct_answer_text(record['answer'], answer_choices)
            
            # Create SAT format record
            sat_record = {
                "database_idx": record['id'],
                "question_type": sat_question_type,
                "question": clean_question,
                "answer_choices": answer_choices,
                "correct_answer": correct_answer_text,
                "img_paths": [f"./data/mmsi-bench/images/{record['id']}_{i}.jpg" 
                             for i in range(record.get('num_images', 1))]
            }
            
            # Group by question type
            if sat_question_type not in questions_by_type:
                questions_by_type[sat_question_type] = []
            questions_by_type[sat_question_type].append(sat_record)
            
        except Exception as e:
            print(f"Warning: Error processing record {record.get('id', 'unknown')}: {e}")
            skipped_records.append(record.get('id', 'unknown'))
            continue
    
    if skipped_records:
        print(f"Skipped {len(skipped_records)} records due to processing errors: {skipped_records[:10]}{'...' if len(skipped_records) > 10 else ''}")
    
    print(f"Found {len(questions_by_type)} question types")
    
    # Set random seed for deterministic shuffling
    random.seed(seed)
    print(f"Using random seed: {seed}")
    
    # Process each question type: shuffle and select subset
    sat_format_data = []
    for question_type, questions in questions_by_type.items():
        print(f"Processing {len(questions)} questions of type '{question_type}'")
        
        # Shuffle questions deterministically
        shuffled_questions = questions.copy()
        random.shuffle(shuffled_questions)
        
        # Select subset if num_questions_per_type is specified
        if num_questions_per_type is not None:
            selected_questions = shuffled_questions[:num_questions_per_type]
            print(f"  Selected {len(selected_questions)} out of {len(questions)} questions")
        else:
            selected_questions = shuffled_questions
            print(f"  Using all {len(questions)} questions")
        
        sat_format_data.extend(selected_questions)
    
    # Sort by database_idx to maintain order
    sat_format_data.sort(key=lambda x: x['database_idx'])
    
    # Save to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(sat_format_data, f, indent=4)
    
    print(f"Successfully processed {len(sat_format_data)} records")
    print(f"Output saved to: {output_path}")
    
    # Print some statistics
    question_types = {}
    for record in sat_format_data:
        qtype = record['question_type']
        question_types[qtype] = question_types.get(qtype, 0) + 1
    
    print("\nFinal question type distribution:")
    for qtype, count in sorted(question_types.items()):
        print(f"  {qtype}: {count}")
    
    # Show pre-filtering statistics
    print(f"\nPre-filtering statistics:")
    print(f"  Original records: {len(mmsi_data)}")
    print(f"  After image filter (≤2 images): {len(filtered_data)}")
    print(f"  Final records: {len(sat_format_data)}")
    
    # Show original vs final counts if per-type filtering was applied
    if num_questions_per_type is not None:
        print(f"\nPer-type filtering results:")
        for qtype in sorted(questions_by_type.keys()):
            prefiltered_count = len(questions_by_type[qtype])
            final_count = question_types.get(qtype, 0)
            print(f"  {qtype}: {prefiltered_count} -> {final_count}")
    
    # Show a sample record
    if sat_format_data:
        print(f"\nSample record (database_idx {sat_format_data[0]['database_idx']}):")
        print(json.dumps(sat_format_data[0], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Process MMSI-Bench data to SAT format")
    parser.add_argument(
        "--mmsi_data_dir",
        type=str,
        default="./data/mmsi-bench/",
        help="Path to MMSI-Bench data directory (default: ./data/mmsi-bench/)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/mmsi-bench/test.json",
        help="Path to output test.json file (default: ./data/mmsi-bench/test.json)"
    )
    parser.add_argument(
        "--num_questions_per_type",
        type=int,
        default=15,
        help="Number of questions to select per question type (default: None, use all questions)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    process_mmsi_bench_data(args.mmsi_data_dir, args.output_file, args.num_questions_per_type, args.seed)


if __name__ == "__main__":
    main()
