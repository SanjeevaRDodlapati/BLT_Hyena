#!/bin/tcsh
# Training job script for Hyena-GLT
# Usage: ./job_scripts/training_job.csh

echo "Starting model training job..."
echo "Timestamp: `date`"

# Set environment variables
setenv PYTHONPATH "/home/sdodl001/BLT_Hyena:$PYTHONPATH"
setenv CUDA_VISIBLE_DEVICES "0,1,2,3"  # Adjust based on available GPUs

# Create output directories
mkdir -p training_output/
mkdir -p logs/

# Run training
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \
  --config configs/basic_training.json \
  > logs/training.log 2>&1

if ($status == 0) then
    echo "Training completed successfully!"
    echo "Model saved to: training_output/"
else
    echo "Training failed with exit code: $status"
    exit 1
endif

echo "Job completed at: `date`"
