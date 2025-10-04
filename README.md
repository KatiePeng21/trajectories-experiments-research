# Trajectories-Experiments-Research

This repository is an independent continuation of our trajectory prediction research originally explored during the HUVTSP program. It is no longer affiliated with Vanguard Defense or any internship. This is a private research project conducted as a group, with the goal of predicting airplane trajectories using the Oslo dataset and systematically comparing different machine learning and deep learning models.

---

## Project Overview

The focus of this project is to investigate a variety of sequence modeling approaches for aircraft trajectory prediction. We are building a reproducible research framework where data, models, experiments, and results are clearly organized and documented.

**Key objectives:**
- Develop a clean data pipeline for the Oslo trajectory dataset.
- Implement and train multiple models (GRU, LSTM, Transformer Encoder, Temporal Convolutional Network, Mamba, etc.).
- Conduct structured experiments with versioned configurations and metrics.
- Compare models using standardized evaluation metrics (MAE, RMSE, geodesic error).
- Visualize predicted vs. actual trajectories in 2D and 3D.

---

## Repository Structure
```
trajectories-experiments-research/
├── data/
├── src/
│ ├── data/
│ ├── models/
│ ├── utils.py
│ ├── train.py
│ └── eval.py
├── experiments/
│ ├── exp_001/...
│ ├── exp_mamba/...
│ └── exp_transformer_encoder/...
├── notebooks/
├── scripts/
├── tests/
└── README.md
```

---

## Setup and Installation

We recommend Python 3.10+ with a virtual environment.

```bash
# clone the repository
git clone https://github.com/<your-username>/trajectories-experiments-research.git
cd trajectories-experiments-research

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows use: .venv\Scripts\activate

pip install -r requirements.txt  

Minimal requirements.txt starter:  
torch  
numpy  
pandas  
scikit-learn  
matplotlib  
tqdm  
pyyaml  

---

## Running a Baseline Example

python -m src.train --model gru --epochs 10 --batch-size 64 --lr 1e-3  

This should:  
- write metrics to experiments/exp_XXX/metrics.json  
- append a row to experiments/runs.csv  
- save the best checkpoint to artifacts/  

---

## .gitignore (important entries)

__pycache__/        *.py[cod]      .ipynb_checkpoints/  
.venv/  venv/  env/  
.DS_Store  Thumbs.db  

# data & heavy outputs  
data/  
artifacts/  
logs/  
wandb/  
outputs/  

# large files  
*.csv  
*.parquet  
*.zip  
*.npz  
*.pt  
*.pth  

---

## Contributing Workflow

- Branches: feat/<thing>, exp/<short-run-name>, fix/<bug>  
- After a run: create experiments/exp_XXX_<tag>/ and update experiments/runs.csv  
- Keep notebooks lightweight; move reusable code into src/  
- Never commit large data or checkpoints (keep in data/ or artifacts/)  

---

## Naming Tips

- Experiments: exp_003_tcn_dilations, exp_012_xfmr_lr1e-4  
- Checkpoints: <model>_<dataset>_<timestamp>.pt  
- Clear commits: feat: add TCN block, exp: GRU vs LSTM bs=128  
