This repository contains code for the ICLR 2026 submission: **Why Transformers Succeed and Fail at Compositional Generalization: Composition Equivalence and Module Coverage**.

Our implementation builds on the code provided by (Ramesh et al., 2024): https://github.com/rahul13ramesh/compositional_capabilities/

To reproduce the results:
1. Generate data for different train-test split strategies. `./bash_scripts/data.sh`
2. Train model for a given train-test split strategy. `./bash_scripts/train_model.sh`
3. Evaluate trained model on a given test distribution. `./bash_scripts/evaluate_model.sh`
   - There is an option to do equivalence class analysis and final layer representation analysis during evaluation. See `./bash_scripts/evaluate_model.sh` for more details.
4. Plot the compositional generalization performance of direct and step-by-step models for absolute and relative positional embeddings. This script generates the plots presented in the paper.
`./bash_scripts/plotting.sh`
5. Genrate TSNE plots (Figures 12 and 13) by running equivalence class and representation analysis during evaluation and then running the following bash script. `./bash_scripts/equivalence_analysis.sh`