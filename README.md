# Adversarial Distance and Membership Risk Auditing

This repository contains the code to replicate the empirical validation and zero-overhead auditing experiments. The scripts evaluate the relationship between training dynamics (Cumulative Sample Loss and Gradient), L2 adversarial vulnerability (d_adv), and Membership Inference Attack (MIA) risk.

## Directory Structure

### 1. validation_experiments/
Contains the core evaluation pipelines for mapping learning trajectories to adversarial distance and privacy leakage across ResNet-18, 34, and 50 architectures.

* train_eval_full.py: Trains target models on the full CIFAR-10/CIFAR-100 datasets across 3 seeds. Tracks CSL/CSG over 200 epochs and evaluates exact ground-truth d_adv using the Carlini & Wagner (C&W) attack.
* train_eval_mia.py: Splits datasets in half to establish in-member configurations. Trains 3 target seeds and 64 shadow models. Computes C&W d_adv, Yeom loss-threshold advantage, and LiRA probabilistic advantage for all in-member samples.
* plot_verification.py: Generates the 10-bin canonical grid plots (Figure 1 replication) comparing CSL, CSG, proxy estimators, adversarial distance, and MIA advantages.
* run_full.sh: Execution script for the full-dataset routine.
* run_mia.sh: Execution script for the MIA routine and subsequent verification plotting.

### 2. adversarial_distance_auditing/
Contains the zero-overhead auditing baselines for ResNet-18 (Appendix C.5).

* pgd_and_d_adv.py: Trains target models over the full 200-epoch horizon, extracting CSL and CSG. Audits the training samples utilizing a 25-step binary search Projected Gradient Descent (PGD) bounded at eps = 6.0 across 3, 5, 10, and 20 iterations, alongside the C&W baseline. Calculates Spearman rank correlations.
* run_auditing_exp.sh: Execution script for the auditing pipeline.

### 3. Extended Experiments
For mislabeled sample detection and duplicate detection experiments, base code from the CSL-Mem repository is used:
git clone https://github.com/DeepakTatachar/CSL-Mem.git

For zero-cost privacy risk evaluation experiments, base code from the Loss Traces repository is used:
git clone https://github.com/imperial-aisp/loss_traces.git

Refer to the dedicated guides for configuration and execution:
* Mislabeled data detection setup: README_mislabeled.md
* Duplicate sample detection setup: README_duplicate.md
* Privacy risk auditing setup: README_privacy.md

Adding our proposed metrics to these existing pipelines required only defining an evaluation function, as their training loops already collect the required per-sample loss and gradient dynamics during standard training without additional computational overhead.

---

## How to Run

Navigate into the respective directories, grant execution permissions to the bash scripts, and execute them.

### Running the Validation Experiments

cd validation_experiments
chmod +x run_full.sh run_mia.sh

# 1. Run full dataset training and C&W evaluation
./run_full.sh

# 2. Run MIA half-dataset evaluation and generate grid plots
./run_mia.sh

Results and plots will be saved to validation_experiments/results/ and validation_experiments/plots/.

### Running the Auditing Baselines

cd adversarial_distance_auditing
chmod +x run_auditing_exp.sh

# Execute the PGD binary search and correlation evaluations
./run_auditing_exp.sh

Results and correlation metrics will be saved to adversarial_distance_auditing/results/ and printed directly to the terminal.
