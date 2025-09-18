# Copilot / AI Agent Instructions for This Repo

> **Purpose**: Help an AI agent become productive fast in this codebase for Grain Process Optimization with ANNs.

## 1) Big picture

* We model a manufacturing process with inputs **X1=Pressure (MPa), X2=Temperature (°C), X3=Time (s)** and outputs **Y1–Y5** (quality metrics). Primary goals: **maximize Y1**, **maximize Y4**, **minimize Y3 & Y5**, and **minimize Y2** (lower priority).
* Phase 1 focuses on **single‑output regression** (pick one Y\*). Phase 2 is **multi‑output regression** (predict all Y1–Y5 jointly) and **multi‑objective optimization** of X1–X3 under bounds.
* Typical stack: Python, pandas, numpy, scikit‑learn (MLPRegressor) or PyTorch/Keras for deeper ANNs, plus opt libraries (nevergrad/scipy).

## 2) Auto‑discovery checklist (run first)

Perform a one‑time repo scan and cache findings:

* Look for: `pyproject.toml` or `requirements.txt`, `Makefile`, `src/**`, `notebooks/**`, `data/**`, `models/**`, `configs/**`, `scripts/**`, `.env`, CI files in `.github/workflows/**`.
* Find AI rules to merge (one glob search): `**/{.github/copilot-instructions.md,AGENT.md,AGENTS.md,CLAUDE.md,.cursorrules,.windsurfrules,.clinerules,.cursor/rules/**,.windsurf/rules/**,.clinerules/**,README.md}`.
* If an older `.github/copilot-instructions.md` exists, **preserve unique guidance**, update paths/commands, and drop deprecated sections.

## 3) Data & schema (project‑specific)

* Default dataset path: `data/Sample Data Sheet.csv` (CSV).
* Expected columns (case‑sensitive): `X1`, `X2`, `X3`, `Y1`, `Y2`, `Y3`, `Y4`, `Y5`.
* Valid ranges used for optimization constraints:

  * X1 ∈ \[400, 600] MPa (in‑range), X2 ∈ \[50, 60] °C (minimize), X3 ∈ \[10, 20] s (minimize).
  * Targets: Maximize Y1 (priority 5), Maximize Y4 (4), Minimize Y3 (3), Minimize Y5 (3), Minimize Y2 (2).
* Standard preprocessing: drop NA, **StandardScaler** for inputs (and optionally outputs for ANN stability), train/val/test split by random or `KFold`.

## 4) Modeling patterns (how we build models here)

* Start with **single‑output** ANN: e.g., `MLPRegressor(hidden_layer_sizes=(64,64), activation='relu', random_state=42)`.
* **Multi‑output** options:

  * scikit‑learn `MultiOutputRegressor(MLPRegressor(...))` (simple, independent heads), or
  * Framework model with **shared trunk + 5 output heads** for correlated targets.
* Metrics to report **per target**: MAE, RMSE, R². Save artifacts to `models/<model_name>/`.
* Repro: set `numpy.random.seed` and framework seeds; log config to `models/<model_name>/config.yaml`.

## 5) Optimization (how we choose X1–X3)

* Define a **weighted objective** using priority weights w = {Y1:5, Y4:4, Y3:3, Y5:3, Y2:2}.

  * Example (minimize): `loss =  -5*ŷ1  -4*ŷ4  +3*ŷ3  +3*ŷ5  +2*ŷ2` (negatives for maximize).
* Search within bounds using `scipy.optimize.minimize`, grid/Latin hypercube, or evolutionary search (e.g., nevergrad). Always **scale/clip** to bounds.
* Return **top‑k candidate settings** and a **pareto.csv** when using multi‑objective optimizers.

## 6) Commands & workflows (run exactly like this)

* Environment (if `pyproject.toml` exists):

  ```bash
  python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
  pip install -U pip
  pip install -e .  # or: pip install -r requirements.txt
  ```
* Train (single output):

  ```bash
  python scripts/train.py --target Y1 --model mlp --hidden 64 64 --epochs 200 --seed 42
  ```
* Evaluate:

  ```bash
  python scripts/eval.py --checkpoint models/mlp_y1/latest.ckpt --test-csv data/Sample\ Data\ Sheet.csv
  ```
* Optimize:

  ```bash
  python scripts/optimize.py --checkpoint models/mlp_multi/latest.ckpt --bounds "X1:400,600;X2:50,60;X3:10,20" --topk 10
  ```
* Notebooks should read from `data/` and write figures to `reports/figures/`.

## 7) Conventions & style (project‑specific)

* Paths are **relative to repo root**; avoid hard‑coding absolute paths.
* Keep **feature order** as `[X1, X2, X3]`; maintain consistent scaler fit from train split.
* Save scalers to `models/<name>/scaler.joblib`; load before inference/optimization.
* Use `configs/*.yaml` for model hyperparams if present; scripts accept CLI flags that override YAML.

## 8) Cross‑component integration

* `scripts/train.py` → writes artifacts (`.pt`/`.pkl`/`.joblib`, `config.yaml`, `metrics.json`).
* `scripts/eval.py` → loads artifacts and computes metrics; writes `reports/metrics/<model>.json`.
* `scripts/optimize.py` → loads trained predictor + scaler; outputs `reports/opt/pareto.csv` and `topk.json`.

## 9) Examples (copy/paste)

* Train multi‑output with independent heads (sklearn):

  ```bash
  python scripts/train.py --target all --model multioutput-mlp --hidden 128 64 --epochs 300
  ```
* Quick LHS search after training:

  ```bash
  python scripts/optimize.py --sampler lhs --samples 1000 --objective weighted --topk 5
  ```

## 10) What the agent should NOT change

* Column names `X1,X2,X3,Y1..Y5` and their meanings.
* Optimization bounds and priority weights unless explicitly instructed.

---

### If an older instruction file exists

* Merge sections that mention **custom commands, folder conventions, or CI steps**; keep them ahead of defaults above.
* Remove stale paths/commands; prefer the examples here if duplicates conflict.

> **Questions for maintainers**: Are the exact script names/paths different? Is there a Makefile/poetry config to prefer? Provide a repo tree snapshot if these defaults don’t match.
