# Trajectories Experiments Research

Research-oriented sandbox for experimenting with deep learning models (GRU, LSTM, Transformer, TCN, etc.) for trajectory prediction.  

---

## Repository Structure

trajectories-experiments-research/  
├─ README.md                         # project overview + how to run  
├─ .gitignore                        # keeps data/models/logs out of git  
├─ requirements.txt                  # Python dependencies  
├─ LICENSE                           # license file  
│  
├─ data/                             # (gitignored) datasets  
│  ├─ raw/                           # original CSV/Parquet/JSON  
│  └─ processed/                     # cleaned/preprocessed tensors/npz  
│  
├─ src/                              # source code (importable package)  
│  ├─ __init__.py                    # marks src as a package  
│  ├─ data/  
│  │   ├─ __init__.py  
│  │   ├─ dataset.py                 # PyTorch Dataset/DataLoader utils  
│  │   └─ preprocess.py              # feature engineering/normalization  
│  ├─ models/  
│  │   ├─ __init__.py                # re-export model classes for clean imports  
│  │   ├─ gru.py                     # GRUModel  
│  │   ├─ lstm.py                    # LSTMModel  
│  │   ├─ transformer.py             # small transformer variants  
│  │   └─ tcn.py                     # temporal conv/TCN models  
│  ├─ train.py                       # training loop entrypoint (CLI)  
│  ├─ eval.py                        # metrics (ADE/FDE/MSE) + eval routines  
│  ├─ utils.py                       # set_seed, checkpoint io, config helpers  
│  └─ visualization.py               # plotting trajectories/curves  
│  
├─ notebooks/                        # Jupyter exploration & demos  
│  ├─ 00_eda.ipynb                   # exploratory data analysis  
│  ├─ 01_baseline_gru.ipynb          # baseline demo (calls src code)  
│  └─ scratch/                       # sandbox notebooks (personal)  
│  
├─ experiments/                      # lightweight, text-based run logs (commit)  
│  ├─ runs.csv                       # append-only summary table of runs  
│  ├─ exp_001_gru_baseline/  
│  │   ├─ config.yaml                # exact hyperparams/paths used  
│  │   ├─ metrics.json               # final metrics (ADE/FDE/MSE/etc.)  
│  │   └─ notes.md                   # brief human notes/observations  
│  └─ exp_002_lstm_bs128/  
│      ├─ config.yaml  
│      ├─ metrics.json  
│      └─ notes.md  
│  
├─ artifacts/                        # large outputs (DO NOT COMMIT)  
│  └─ gru_baseline.pt                # example checkpoint (gitignored)  
│  
├─ logs/                             # tensorboard/wandb/raw logs (gitignored)  
│  
├─ scripts/                          # small, runnable helpers (not imported)  
│  ├─ prepare_data.py                # raw → processed converter  
│  └─ run_experiment.sh              # convenience launcher  
│  
└─ tests/                            # optional unit tests (pytest)  
    ├─ test_dataset.py               # shapes/dtypes/lengths sanity checks  
    └─ test_models.py                # forward pass/param count checks  

---

## Folder & File Guide

### Root
- README.md — Structure, setup, usage, and contribution notes.  
- .gitignore — Prevents committing bulky or generated files (data, logs, artifacts).  
- requirements.txt — Python dependencies for reproducible installs.  
- LICENSE — (Optional) MIT/Apache if sharing publicly.  

### data/ (gitignored)  
- raw/ — Original datasets (CSV/Parquet/JSON). Keep untouched for provenance.  
- processed/ — Cleaned/feature-engineered arrays/tensors ready for training.  

### src/ (importable project code)  
- data/dataset.py — PyTorch Dataset + DataLoader utilities.  
- data/preprocess.py — Feature engineering, normalization, splitting.  
- models/*.py — One file per architecture (GRUModel, LSTMModel, etc.).  
- train.py — CLI training entrypoint; reads configs/args, saves metrics/ckpts.  
- eval.py — Evaluation metrics (ADE/FDE/MSE) and workflows.  
- utils.py — set_seed, checkpoint save/load, config parsing, misc helpers.  
- visualization.py — Plotting functions (trajectories, learning curves, histograms).  

### notebooks/  
- 00_eda.ipynb — Quick data sanity checks and basic plots.  
- 01_baseline_gru.ipynb — Minimal baseline demo that calls into src/.  
- scratch/ — Personal experiments; keep them out of PRs unless cleaned.  

### experiments/  
- runs.csv — Append-only overview of all runs (timestamp, model, hyperparams, metrics, notes).  
- exp_XXX_* — One folder per meaningful run:  
  - config.yaml — Exact settings used.  
  - metrics.json — Final numbers (ADE/FDE/MSE/etc.).  
  - notes.md — Short notes: what changed, observations, next steps.  

### artifacts/ (gitignored)  
- Model checkpoints (*.pt, *.pth), cached tensors, prediction arrays.  

### logs/ (gitignored)  
- TensorBoard runs, Weights & Biases directories, raw training logs.  

### scripts/  
- prepare_data.py — Convert raw → processed.  
- run_experiment.sh — Bash one-liner to launch training.  

### tests/ (optional)  
- test_dataset.py — Checks dataset shapes/lengths.  
- test_models.py — Ensures model forward passes run correctly.  

---

## Getting Started

Clone repo and set up environment:

git clone git@github.com:KatiePeng21/trajectories-experiments-research.git  
cd trajectories-experiments-research  

python -m venv .venv  
# Windows  
. .venv\Scripts\Activate.ps1  
# macOS/Linux  
# source .venv/bin/activate  

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
