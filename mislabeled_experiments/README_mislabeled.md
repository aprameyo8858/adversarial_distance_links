# Mislabeled Sample Detection Experiment

This guide describes how to set up and execute the mislabeled sample detection experiments using the base implementation from CSL-Mem.

## Step 1: Clone the Repository
git clone https://github.com/DeepakTatachar/CSL-Mem.git
cd CSL-Mem

## Step 2: Environment and Storage Setup
1. Ensure your Python environment has PyTorch, TensorFlow, and MinIO installed.
2. Start the local MinIO storage container via Docker:

docker pull minio/minio
docker run -p 9000:9000 -p 9001:9001 --name minio \
    -e "MINIO_ROOT_USER=<your-access-key>" \
    -e "MINIO_ROOT_PASSWORD=<your-secret-key>" \
    minio/minio server /data --console-address ":9001"

3. Configure config.json in the root directory with your local dataset and model checkpoint directories.
4. Configure credentials.json with the corresponding MinIO credentials and endpoint URL.
5. Place the required FZ influence matrix checkpoints under analysis_checkpoints/dataset.

## Step 3: Run Training and Scoring
Execute the mislabeled experiment pipeline:

sh ./scripts/mislabeled_exps.sh

This script trains the baseline models, k-fold confident learning models, and SSFT models across noise ratios and seeds on CIFAR-10 and CIFAR-100, recording per-epoch scoring statistics.

## Metric Evaluation
Our metric is integrated directly as an evaluation method alongside existing baseline techniques. Since the training routine natively records the necessary per-sample training dynamics, incorporating our metric only requires adding a scoring function without incurring extra training passes or retraining costs.
