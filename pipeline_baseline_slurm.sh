#!/bin/bash

#SBATCH --job-name=mindjourney_baseline
#SBATCH --gres=gpu:80gb:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/j/jhas/slurm-%j.out
#SBATCH -e /network/scratch/j/jhas/slurm-%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "----------------------------------------"

# Load modules
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

# Azure OpenAI configuration (set your credentials via environment variables or uncomment and set below)
# export AZURE_OPENAI_API_KEY="your_api_key"
# export AZURE_OPENAI_ENDPOINT="https://your-endpoint.cognitiveservices.azure.com/"

# 1. Copy project files to compute node's temporary directory
echo "Copying project files to $SLURM_TMPDIR..."
cp -r $SLURM_SUBMIT_DIR/* $SLURM_TMPDIR/

# 2. Set up virtual environment on compute node
echo "Setting up virtual environment on compute node..."
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Prepare data on compute node
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
if [ ! -f "data/test.json" ]; then
    echo "ERROR: Data preparation failed. data/test.json not found."
    exit 1
fi

echo "Data preparation successful. Found data/test.json"
echo "Number of questions in test set: $(python -c "import json; print(len(json.load(open('data/test.json'))))")"

# Verify installation
echo "Verifying key package installations..."
python -c "import openai, transformers, numpy, decord; print('Key packages imported successfully')"

# 5. Configuration parameters
num_questions=150  # Reduced for testing and space constraints
scaling_strategy="spatial_beam_search"
question_type="None"
vlm_model_name="OpenGVLab/InternVL3-14B"
vlm_qa_model_name=None
max_images=2
dataset_type="test"
input_dir="data/mmsi-bench"

sanitized_model_name=$(echo "$vlm_model_name" | sed 's|/|_|g')
sanitized_dataset_type=$(echo "$input_dir" | sed 's|/|_|g')
output_dir="results/results_baseline_${sanitized_dataset_type}_${sanitized_model_name}_${num_questions}_${max_images}"



# Create output directory on compute node
mkdir -p $output_dir

# 6. Build and execute the command
cmd="python pipelines/pipeline_baseline.py \
  --vlm_model_name=$vlm_model_name \
  --vlm_qa_model_name=$vlm_qa_model_name \
  --num_questions $num_questions \
  --output_dir $output_dir \
  --input_dir $input_dir \
  --question_type None \
  --max_images $max_images \
  --max_tries_gpt 5 \
  --split test \
  --num_question_chunks 1 \
  --question_chunk_idx 0 \
  --task 'img2trajvid_s-prob' \
  --replace_or_include_input True \
  --cfg 4.0 \
  --guider 1 \
  --L_short 576 \
  --num_targets 8 \
  --use_traj_prior True \
  --chunk_strategy 'interp'"

echo "Running command: $cmd"
echo "----------------------------------------"

# Execute the command
eval $cmd

# 7. Copy results back to scratch space
if [ $? -eq 0 ]; then
    echo "Job completed successfully!"
   
else
    echo "Job failed with exit code $?"
fi


echo "End Time: $(date)"
echo "-------------------- Finished executing script --------------------"
