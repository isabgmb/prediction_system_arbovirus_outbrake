# Dengue Forecasting — Python Port

Python conversion of the MATLAB pipeline originally described in:

> *Data-Driven Computational Intelligence Applied to Dengue Outbreak Forecasting:
> a case study at the scale of the city of Natal, RN-Brazil*

---

## Folder structure

```
dengue-forecasting/
├── data/
│   ├── raw/          ← put all original .mat and .xlsx files here
│   ├── processed/    ← cleaned CSVs (generated automatically)
│   └── models/       ← saved model weights & eval results (.npz)
├── notebooks/
│   └── 02_model_eval.ipynb   ← interactive evaluation & plots
├── src/
│   ├── utils.py      ← data loaders + metric functions
│   ├── train.py      ← LSTM training (replaces Run1DLSTM_01.m)
│   └── evaluate.py   ← plotting & evaluation (replaces ModelsPerform_1DLSTM_01.m)
├── outputs/          ← saved figures
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Required data files

Place the following files inside **`data/raw/`**:

| File | Source |
|---|---|
| `Models_1DLSTM_2022_02_04_EvalResults.mat` | Pre-computed MATLAB results |
| `Rearranged_Data.mat` | EDI / Ovitrap data |
| `Dados_Modelagem.xlsx` | Dengue incidence (sheet 4) |
| `DADOS GERAIS_01.xlsx` | Socioeconomic data (sheet 2) |

---

## Running the code

### Option A — Jupyter notebook (recommended for exploration)

```bash
jupyter notebook notebooks/02_model_eval.ipynb
```

### Option B — Evaluate pre-trained results (no GPU needed)

```bash
python src/evaluate.py
```

Figures are saved to `outputs/`.

### Option C — Re-train the LSTM models from scratch

```bash
python src/train.py
```

> ⚠️ Training 10 models × 30 repetitions takes several hours on CPU.
> A GPU is strongly recommended. TensorFlow will use one automatically if available.

---

## MATLAB → Python mapping

| MATLAB file | Python equivalent |
|---|---|
| `Run1DLSTM_01.m` | `src/train.py` |
| `ModelsPerform_1DLSTM_01.m` | `src/evaluate.py` + `notebooks/02_model_eval.ipynb` |
| `ComputeRMSE` (nested fn) | `src/utils.compute_rmse()` |
| `ComputeR` (nested fn) | `src/utils.compute_r()` |
| `TrainLSTM` (nested fn) | `src/train.train_lstm()` |

---

## License

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
