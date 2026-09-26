# Duplicate Sample Detection Experiment

This guide describes how to set up and execute the duplicate sample detection experiments using the base framework from CSL-Mem.

## Step 1: Clone the Repository
git clone https://github.com/DeepakTatachar/CSL-Mem.git
cd CSL-Mem

## Step 2: Environment and Storage Setup
1. Verify that PyTorch, TensorFlow, and MinIO are available in your environment.
2. Launch the MinIO container:

docker pull minio/minio
docker run -p 9000:9000 -p 9001:9001 --name minio \
    -e "MINIO_ROOT_USER=<your-access-key>" \
    -e "MINIO_ROOT_PASSWORD=<your-secret-key>" \
    minio/minio server /data --console-address ":9001"

3. Create config.json and credentials.json in the root folder with the appropriate directory paths and MinIO access keys.
4. Ensure the influence checkpoints are placed in analysis_checkpoints/dataset.

## Step 3: Run Duplicate Experiments
Execute the duplicate training and evaluation script:

sh ./scripts/duplicate_exps.sh

This script trains SSFT models, confident learning models, and standard models across partitions on CIFAR-10 and CIFAR-100 duplicate datasets while tracking per-epoch scores.

## Metric Evaluation
Our metric is evaluated alongside the standard baseline methods directly during the scoring pass. Because the underlying training process already generates the full sequence of loss and gradient states, our metric was added simply as a functional call, requiring no extra compute or architectural modifications.
