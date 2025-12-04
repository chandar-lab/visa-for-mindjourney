#!/bin/bash
#SBATCH --job-name=mmsi_download
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /network/scratch/j/jhas/mmsi_download-%j.out
#SBATCH -e /network/scratch/j/jhas/mmsi_download-%j.err

# MMSI-Bench Dataset Download Script
# This script downloads the MMSI-Bench dataset using the HuggingFace datasets library


# Create logs directory if it doesn't exist
mkdir -p logs

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
echo "Activating venv environment..."

# Azure OpenAI configuration (set your credentials via environment variables or uncomment and set below)
# export AZURE_OPENAI_API_KEY="your_api_key"
# export AZURE_OPENAI_ENDPOINT="https://your-endpoint.cognitiveservices.azure.com/"

# Hugging Face authentication (set your token via environment variable or uncomment and set below)
# export HUGGINGFACE_HUB_TOKEN="your_hf_token"
# Alternative: export HF_TOKEN="your_hf_token"


source venv/bin/activate


# Check if we're in the right directory
if [ ! -f "utils/mmsi-bench-download.py" ]; then
    echo "Error: mmsi-bench-download.py not found in utils/ directory"
    echo "Current directory: $(pwd)"
    echo "Contents: $(ls -la)"
    exit 1
fi

# Check Python and required packages
echo "Checking Python environment..."
python --version

# Check if required packages are installed
echo "Checking required packages..."
python -c "import pandas, datasets" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install pandas datasets
fi

# Create data directory
echo "Creating data directory..."
mkdir -p data/mmsi-bench

# Run the download script
echo "Starting dataset download..."
python utils/mmsi-bench-download.py --output_dir data/mmsi-bench/

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Dataset download completed successfully!"
    
    # Display summary
    echo "Dataset summary:"
    echo "Images directory: $(ls -la data/mmsi-bench/images/ | wc -l) files"
    echo "Metadata files: $(ls -la data/mmsi-bench/metadata_*.json 2>/dev/null | wc -l) files"
    echo "Total size: $(du -sh data/mmsi-bench/)"
    
    # Verify a few sample files
    echo "Verifying sample files..."
    if [ -f "data/mmsi-bench/mmsi_bench_metadata.csv" ]; then
        echo "✓ Metadata CSV file created"
        echo "First few lines of metadata:"
        head -5 data/mmsi-bench/mmsi_bench_metadata.csv
    fi
    
    if [ -f "data/mmsi-bench/mmsi_bench_metadata.json" ]; then
        echo "✓ Metadata JSON file created"
    fi
    
    # Check image files
    if [ -d "data/mmsi-bench/images" ]; then
        echo "✓ Images directory created"
        echo "Sample image files:"
        ls data/mmsi-bench/images/ | head -5
    fi
    
else
    echo "Error: Dataset download failed!"
    exit 1
fi

echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
