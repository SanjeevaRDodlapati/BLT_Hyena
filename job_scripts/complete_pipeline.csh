#!/bin/tcsh
# Complete pipeline job script for Hyena-GLT
# Usage: ./job_scripts/complete_pipeline.csh /path/to/your/data.fasta

echo "Starting complete Hyena-GLT pipeline..."
echo "Timestamp: `date`"

# Check if data path argument is provided
if ($#argv != 1) then
    echo "Usage: $0 <input_data_path>"
    echo "Example: $0 /path/to/your/genome_sequences.fasta"
    exit 1
endif

set INPUT_DATA = $1
echo "Input data: $INPUT_DATA"

# Set environment variables
setenv PYTHONPATH "/home/sdodl001/BLT_Hyena:$PYTHONPATH"
setenv CUDA_VISIBLE_DEVICES "0"

# Create directories
mkdir -p processed_data/
mkdir -p training_output/
mkdir -p evaluation_results/
mkdir -p logs/

echo "Step 1: Data Preprocessing..."
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --input $INPUT_DATA \
  --output processed_data/pipeline_data/ \
  --task sequence_classification \
  --max-length 1024 \
  --format hdf5 \
  --progress > logs/preprocessing.log 2>&1

if ($status != 0) then
    echo "Preprocessing failed!"
    exit 1
endif

echo "Step 2: Model Training..."
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \
  --data processed_data/pipeline_data/ \
  --output training_output/pipeline_model/ \
  --model small \
  --epochs 5 \
  --batch-size 16 > logs/training.log 2>&1

if ($status != 0) then
    echo "Training failed!"
    exit 1
endif

echo "Step 3: Model Evaluation..."
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.eval \
  --model training_output/pipeline_model/final_model.pt \
  --data processed_data/pipeline_data/test.hdf5 \
  --output evaluation_results/pipeline_model/ > logs/evaluation.log 2>&1

if ($status != 0) then
    echo "Evaluation failed!"
    exit 1
endif

echo "Pipeline completed successfully!"
echo "Results available in:"
echo "  - Processed data: processed_data/pipeline_data/"
echo "  - Trained model: training_output/pipeline_model/"
echo "  - Evaluation results: evaluation_results/pipeline_model/"
echo "Timestamp: `date`"
