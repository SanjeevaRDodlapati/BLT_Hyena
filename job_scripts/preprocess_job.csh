#!/bin/tcsh
# Preprocessing job script for Hyena-GLT
# Usage: ./job_scripts/preprocess_job.csh

echo "Starting data preprocessing job..."
echo "Timestamp: `date`"

# Set environment variables
setenv PYTHONPATH "/home/sdodl001/BLT_Hyena:$PYTHONPATH"
setenv CUDA_VISIBLE_DEVICES "0"

# Create output directories
mkdir -p processed_data/
mkdir -p logs/

# Run preprocessing with your data path
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --input data/your_genome_sequences.fasta \
  --output processed_data/genome_classification/ \
  --task sequence_classification \
  --max-length 1024 \
  --min-length 50 \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1 \
  --format hdf5 \
  --compress \
  --num-workers 8 \
  --log-level INFO \
  --progress > logs/preprocessing.log 2>&1

if ($status == 0) then
    echo "Preprocessing completed successfully!"
    echo "Output saved to: processed_data/genome_classification/"
else
    echo "Preprocessing failed with exit code: $status"
    exit 1
endif

echo "Job completed at: `date`"
