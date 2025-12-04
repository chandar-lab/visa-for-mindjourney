#!/bin/bash

# Enhanced pipeline with VLM verifier for Video COT approach
# This script runs the verifier-enabled version while keeping the original as baseline

export WORLD_MODEL_TYPE="svc"
export QUESTION_DATABASE_TYPE="SAT"

# Run with verifier enabled (default)
echo "Running Spatial VQA Pipeline with VLM Verifier..."
python pipelines/pipeline_svc_scaling_spatial_beam_search_with_verifier.py \
    --enable_verifier \
    --verification_threshold 0.7 \
    --input_dir data/SAT \
    --output_dir outputs/svc_with_verifier \
    --vlm_model_name "gpt-4o" \
    --vlm_qa_model_name "None" \
    --max_steps_per_question 3 \
    --max_images 1 \
    --num_questions 10 \
    --question_type "None" \
    --split "val" \
    --scaling_strategy "spatial_beam_search" \
    --fixed_forward_magnitudes "0.25,0.5,0.75" \
    --fixed_rotation_magnitudes "10,20,30" \
    --sampling_interval_angle 10 \
    --sampling_interval_meter 0.25 \
    --max_turn_angle 60 \
    --max_forward_distance 1.5 \
    --num_beams 3 \
    --num_top_candidates 5 \
    --exploration_score_threshold 5 \
    --helpful_score_threshold 6 \
    --max_tries_gpt 3 \
    --max_inference_batch_size 4 \
    --num_frames 4 \
    --frame_interval 3 \
    --task "video" \
    --replace_or_include_input "replace" \
    --cfg 1.0 \
    --guider "none" \
    --L_short 72 \
    --num_targets 1 \
    --use_traj_prior True \
    --chunk_strategy "none"

echo "Verifier-enabled pipeline completed!"
