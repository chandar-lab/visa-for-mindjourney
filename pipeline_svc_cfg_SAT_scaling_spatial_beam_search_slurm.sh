#!/bin/bash

#SBATCH --job-name=mindjourney_svc_sat_scaling
#SBATCH --gres=gpu:80gb:2
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/j/jhas/slurm-%j.out
#SBATCH -e /network/scratch/j/jhas/slurm-%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "----------------------------------------"

# Load modules (uncomment and modify as needed for your cluster)
module load python/3.10
module load cuda/11.8
module load gcc/9.3.0

# Set up environment
export WORLD_MODEL_TYPE="svc"
export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0,1

# Print CUDA information for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs:"
nvidia-smi --list-gpus 2>/dev/null || echo "nvidia-smi not available"

# CUDA environment variables to help with attention and memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# Multiprocessing and CUDA context sharing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Disable multiprocessing for CUDA compatibility (uncomment if needed)
# export CUDA_LAUNCH_BLOCKING=1

# Azure OpenAI configuration (set your credentials via environment variables or uncomment and set below)
# export AZURE_OPENAI_API_KEY="your_api_key"
# export AZURE_OPENAI_ENDPOINT="https://your-endpoint.cognitiveservices.azure.com/"

# Hugging Face authentication (set your token via environment variable or uncomment and set below)
# export HF_TOKEN="your_hf_token"
# Alternative: export HUGGINGFACE_HUB_TOKEN="your_hf_token"


# 1. Copy project files to compute node's temporary directory
echo "Copying project files to $SLURM_TMPDIR..."
cp -r $SLURM_SUBMIT_DIR/* $SLURM_TMPDIR/

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install requirements
echo "Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directory and prepare data
echo "Preparing data on compute node..."
mkdir -p data

# Check if data already exists, if not generate it
if [ ! -f "data/test.json" ]; then
    echo "Generating test data from SAT dataset..."
    python utils/data_process.py --output_dir ./data --split test 
else
    echo "Data already exists, skipping data generation..."
fi

# Verify data exists
if [ -f "data/test.json" ]; then
    echo "Data preparation successful. Found data/test.json"
    echo "Number of questions in test set: $(python -c "import json; print(len(json.load(open('data/test.json'))))")"
else
    echo "ERROR: Data preparation failed. data/test.json not found."
    exit 1
fi

# Configuration parameters
num_questions=150
scaling_strategy="spatial_beam_search"
question_type="None"
vlm_model_name="OpenGVLab/InternVL3-14B"
vlm_qa_model_name=None # pass None will be interpreted as "None" anyway; None means qa_vlm_model_name is same as vlm_model_name
helpful_score_threshold=8
exploration_score_threshold=8
max_images=2
max_steps=2 # Only 1 step for test runs
dataset_type="test" # choose from "val", "test"
input_dir="data"

# Flag to control --no_answer_choices
no_answer_choices=false
compute_uncertainty=false

# Flag to control filtering method (threshold-based vs top-k)
use_top_k_filtering=true

# Top-k parameter for filtering (only used when use_top_k_filtering=true)
top_k=1

# CFG scale value (must match the --cfg value in the command below)
cfg_scale=4.0

# Build output directory name
sanitized_model_name=$(echo "$vlm_model_name" | sed 's|/|_|g')
sanitized_dataset_type=$(echo "$input_dir" | sed 's|/|_|g')
# Replace decimal point with underscore for directory name compatibility
sanitized_cfg_scale=$(echo "$cfg_scale" | sed 's|\.|_|g')
output_dir="results/mj_svc_cfg_v2_cfg${sanitized_cfg_scale}_${sanitized_dataset_type}_${sanitized_model_name}_${num_questions}_${max_steps}_${exploration_score_threshold}_${helpful_score_threshold}_${max_images}"
# Add _no_answer_choices suffix if flag is enabled
if [ "$no_answer_choices" = true ]; then
    output_dir="${output_dir}_no_answer_choices"
fi

# Add _top_k_filtering suffix if flag is enabled
if [ "$use_top_k_filtering" = true ]; then
    output_dir="${output_dir}_top_${top_k}_filtering"
fi


# Build the command
cmd="python pipelines/pipeline_svc_scaling_spatial_beam_search_question_conditioning.py \
  \
  --vlm_model_name=$vlm_model_name \
  --vlm_qa_model_name=$vlm_qa_model_name \
  --num_questions $num_questions \
  --output_dir $output_dir \
  --input_dir $input_dir \
  --scaling_strategy $scaling_strategy \
  --question_type $question_type \
  --helpful_score_threshold $helpful_score_threshold \
  --exploration_score_threshold $exploration_score_threshold \
  --max_images $max_images \
  --sampling_interval_angle 9 \
  --sampling_interval_meter 0.25 \
  --fixed_rotation_magnitudes 27 \
  --fixed_forward_magnitudes 0.75 \
  --max_steps_per_question $max_steps \
  --num_top_candidates 6 \
  --num_beams 3 \
  --max_tries_gpt 4 \
  --num_frames 9 \
  --frame_interval 3 \
  --max_inference_batch_size 1 \
  --split $dataset_type \
  --num_question_chunks 1 \
  --question_chunk_idx 0 \
  --task "img2trajvid_s-prob" \
  --replace_or_include_input True \
  --cfg $cfg_scale \
  --guider 1  \
  --L_short 576  \
  --num_targets 8  \
  --use_traj_prior True --qa-cfg-guidance \
  --qa-cfg-scale 0.4 \
  --qa-cfg-steps "0,1,2" \
  --device_svc "cuda:0" \
  --device_vae "cuda:0" \
  --device_conditioner "cuda:1" \
  --chunk_strategy "interp""

# Add --no_answer_choices flag conditionally
if [ "$no_answer_choices" = true ]; then
    cmd="$cmd  --no_answer_choices"
fi

# Add --use_top_k_filtering flag conditionally
if [ "$use_top_k_filtering" = true ]; then
    cmd="$cmd  --use_top_k_filtering --top_k $top_k"
fi

if [ "$compute_uncertainty" = true ]; then
    cmd="$cmd  --compute_uncertainty"
fi

echo "Running command: $cmd"
echo "----------------------------------------"

# Execute the command
eval $cmd

# Check exit status
if [ $? -eq 0 ]; then
    echo "Job completed successfully!"
    

else
    echo "Job failed with exit code $?"
fi

echo "End Time: $(date)"
echo "-------------------- Finished executing script --------------------"
