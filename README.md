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

# install dependencies
pip install -r requirements.txt
```

---

## Models Explored
- GRU (baseline recurrent sequence model)
- LSTM (classic recurrent model)
- Transformer Encoder (attention-based sequence model)
- Temporal Convolutional Network (TCN)
- Mamba (state space model)
- Additional models under development

---

## Experiment Tracking

Each experiment lives under `experiments/` but the structure is **not yet fully standardized**.  

- Some experiments (e.g. `exp_001/`) follow the full format:
  - `config.yaml` — configuration settings
  - `metrics.json` — performance metrics
  - `notes.md` — qualitative observations
- Other experiments are simpler and currently just contain:
  - `train.py` / `test.py` — experiment-specific scripts

**Note:** Standardization is a work in progress. The goal is for every experiment to eventually include config, metrics, and notes for reproducibility.

---

## Acknowledgements
- Oslo Dataset for providing real-world aircraft trajectories.
- Original trajectory prediction work conducted during HUVTSP (Harvard Undergraduate Ventures TECH Summer Program).

This repository represents independent group research and is not affiliated with Vanguard Defense or any internship program.

