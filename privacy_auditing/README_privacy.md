# Zero-Cost Privacy Risk Evaluation

This guide outlines the setup for evaluating membership inference risk and privacy leakage using the base pipeline from Loss Traces.

## Step 1: Clone and Install
Clone the repository and install it in editable mode:

git clone https://github.com/imperial-aisp/loss_traces.git
cd loss_traces
pip install -e .

## Step 2: Path Configuration
Default data and artifact storage paths are located in src/loss_traces/config.py. You can override these defaults by defining a src/loss_traces/config_local.py file specifying custom STORAGE_DIR and DATA_DIR locations.

## Step 3: Integrating Loss Dynamic Tracking
To collect learning dynamics without overhead during training, instantiate the classification criterion with unreduced loss:

criterion = nn.CrossEntropyLoss(reduction="none")

Record per-sample losses from the batch output before invoking .mean().backward(). When data augmentation is enabled, measure non-augmented per-sample losses via a single forward pass over the training set at each epoch boundary.

## Step 4: Run Target and Shadow Model Pipeline
Execute the full attack pipeline to train the target model, shadow models, and compute membership inference attack scores (such as LiRA, AttackR, and RMIA):

python -m loss_traces.run_attack_pipeline \
    --exp_id wrn28-2_CIFAR10 \
    --arch wrn28-2 \
    --dataset CIFAR10 \
    --n-shadows 256 \
    --full \
    --gpu :0

## Metric Evaluation
Evaluating our privacy auditing metric in this pipeline requires only adding a function to process the recorded loss traces. Because the pipeline already logs per-sample loss trajectories, our metric can be computed directly from the saved logs without training auxiliary shadow models.
