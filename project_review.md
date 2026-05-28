# TATOO Project Review

Generated: 2026-05-27

---

## 1. Core Architecture

TATOO (Tablet-based Assessment Tool for Touchscreen Operations) is a research pipeline that extracts digital biomarkers from tablet-based motor and cognitive assessments, then uses machine learning to classify participants by age group and detect visual impairment. The codebase is organized into two independent pipelines.

### Clinical Pipeline (`scripts/`)

Operates on structured clinical datasets (`data/Children_data.xlsx`, `data/Eldery_data.xlsx`) covering results from 20 tablet-based tests (T1–T20) across children and elderly cohorts.

```
data/*.xlsx
  └─ 01_data_cleaning_and_preprocessing.py   # missing values, dominant-hand filtering
      └─ 02_eda_part1_tables.py               # summary statistics
          └─ 03_eda_part2_correlations.py     # correlation analysis
              └─ 04_feature_selection_pipeline.py  # statistical feature selection
                  └─ 05_supervised_ml.py       # Random Forest + XGBoost + GridSearchCV
                      └─ 06_clustering/        # HDBSCAN, imputed, targeted, top-4 scenarios
                          └─ 07_clinical_analysis/  # profiling, effect sizes, presentation figures
```

Feature metadata is captured in `data/Children_feature_metadata.json` and `data/Eldery_feature_metadata.json`, which record significant variables (51 for children), confounders (Age, Frequency_Tablet_Use), skewed variables requiring log transforms, highly correlated pairs, and redundant features to drop.

Outputs land in `results/clustering_results/{Children,Eldery}/` and `figures/`.

### Gaze Pipeline (`gaze_analysis/scripts/`)

Operates on raw sensor recordings (NPZ compressed archives containing gaze X/Y in smoothed cm and touch X/Y at 30 Hz) from 6 interactive games: TouchIt, CornerIt, DoubleTapIt, PinchIt, SlideIt, DragIt.

```
gaze_analysis/data/raw/<participant>/
  └─ 01_organize_new_data.py       # regex-based file routing into per-participant folders
      ├─ 02_peek_npz.py            # utility: inspect NPZ array structure
      ├─ 03_calculate_sample_rate.py  # utility: detect time units, compute Hz
      └─ 04_sync_and_move.py       # extract arrays → DataFrames → merge_asof(34ms) → synced CSV
          ├─ 05_tapping_features.py    # TouchIt, CornerIt, DoubleTapIt motor features
          ├─ 06_dragging_features.py   # PinchIt, SlideIt, DragIt kinematics + jerk
          ├─ 07_gaze_features.py       # I-VT fixation/saccade detection (20 cm/s threshold)
          └─ 08_sync_features.py       # eye-hand distance, latency, directional alignment
              └─ 09_merge_matrices.py  # outer-join all features + attach demographics
                  ├─ 10_explore_data.py        # EDA, heatmaps, missing data audit
                  ├─ 10b_explore_data.py       # violin plots variant
                  ├─ 10c_random_forest.py      # RF feature importance with filtering
                  ├─ 11_train_model.py          # XGBoost baseline (5-fold stratified CV)
                  ├─ 11b_imbalance_data_training.py  # class-weighted XGBoost
                  ├─ 11c_confusion_matrix.py   # per-patient cross_val_predict heatmap
                  ├─ 12_t-SNE.py               # t-SNE at 4 perplexities (2,5,10,20)
                  └─ 13_categorized_participants.py  # demographics bar chart
```

Outputs land in `gaze_analysis/results/` and `gaze_analysis/data/processed/`.

---

## 2. Dependencies

No `requirements.txt`, `environment.yml`, `setup.py`, or `pyproject.toml` exists anywhere in the repository. The following libraries are required, inferred from script imports:

| Library | Used in |
|---|---|
| `numpy` | All gaze pipeline scripts |
| `pandas` | All gaze pipeline scripts |
| `scipy` | Sync features (cross-correlation) |
| `scikit-learn` | Feature scaling, t-SNE, RF, StratifiedKFold, cross_val_predict |
| `xgboost` | Scripts 11, 11b, 11c |
| `matplotlib` | All visualization scripts |
| `seaborn` | Heatmaps and violin plots |
| `hdbscan` | `scripts/06_clustering/hdbscan_clustering.py` |

No versions are pinned. Reproducing the environment requires manual inspection of each script.

---

## 3. Data Flow

### Gaze Pipeline (detailed)

```
Raw NPZ files (gaze_x, gaze_y @ ~30Hz; touch_x, touch_y @ 30Hz)
        │
        ▼ Script 04: merge_asof(tolerance=34ms) → 30Hz unified timeseries
<game>_synced_timeseries.csv
  ├─ gaze_x_cm, gaze_y_cm (raw value × 0.01)
  ├─ touch_x_raw, touch_y_raw
  └─ timestamp (ms)
        │
        ├─▶ Script 05 (tapping): Accuracy_Ratio, Flight_Touch_Ratio,
        │     Pressure_{Low,Med,High}_Pct, Pressure_Jitter_Stationary
        │     → <game>_ML_Features.csv
        │
        ├─▶ Script 06 (dragging): Mean_Action_Duration_sec,
        │     Spatial_Jerk_Magnitude (cm/s³), stroke-level kinematics
        │     → <game>_ML_Features.csv
        │
        ├─▶ Script 07 (gaze, I-VT): Fixation_Count, Mean_Fixation_Duration_sec,
        │     Saccade_Frequency_Hz, Mean_Saccadic_Amplitude_cm,
        │     Peak_Saccadic_Velocity_cm_s, Gaze_In_Out_Ratio
        │     → <game>_Gaze_Features.csv
        │
        └─▶ Script 08 (sync): Mean_Eye_Hand_Distance_cm,
              True_Eye_Hand_Latency_sec (dragging only),
              Directional_Velocity_Alignment (dragging only)
              → <game>_Sync_Features.csv
                    │
                    ▼ Script 09: outer merge on Patient_ID + demographicsv2.csv
              MASTER_<game>.csv
                (all features + Age + Target_Age_Old + Target_Visual_Impaired)
                    │
                    ├─▶ Scripts 10/10b/10c: EDA, correlation heatmaps, RF importance
                    ├─▶ Scripts 11/11b/11c: XGBoost classification, per-patient matrix
                    └─▶ Script 12: t-SNE dimensionality reduction (4 perplexities)
```

### Clinical Pipeline (high-level)

```
data/{Children,Eldery}_data.xlsx
        │
        ▼ Script 01: cleaning, dominant-hand filtering, test prefix normalization (T1–T20)
*_Cleaned_DominantOnly.csv
        │
        ▼ Scripts 02–03: EDA (tables + correlations)
        │
        ▼ Script 04: feature selection (metadata JSON → significant_vars)
*_Final_ML_Ready.csv
        │
        ├─▶ Script 05: supervised ML (RF + XGBoost, GridSearchCV)
        └─▶ Script 06: clustering (HDBSCAN) → Script 07: clinical profiling
```

---

## 4. Improvement Suggestions

### 1. Add Environment Reproducibility (Priority: High)

**Problem:** No dependency specification file exists. Any collaborator must reverse-engineer the environment from import statements. Version drift between numpy, scikit-learn, or xgboost can silently change numerical results.

**Fix:** Add two files to the repository root:

```
# requirements.txt  (pinned from active tatoo conda env)
numpy==2.2.6
pandas==2.3.3
scipy==1.15.2
scikit-learn==1.7.2
xgboost==3.2.0
matplotlib==3.10.8
seaborn==0.13.2
hdbscan==0.8.41
imbalanced-learn==0.14.1
statsmodels==0.14.6
openpyxl==3.1.5
```

```yaml
# environment.yml  (pinned from active tatoo conda env, python 3.10)
name: tatoo
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10.19
  - numpy=2.2.6
  - pandas=2.3.3
  - scipy=1.15.2
  - scikit-learn=1.7.2
  - xgboost=3.2.0
  - matplotlib=3.10.8
  - seaborn=0.13.2
  - hdbscan=0.8.41
  - imbalanced-learn=0.14.1
  - statsmodels=0.14.6
  - openpyxl=3.1.5
```

To reproduce: `conda env create -f environment.yml` then `conda activate tatoo`.

---

### 2. Centralize Configuration and Fix Age Threshold Inconsistency (Priority: High)

**Problem:** Constants are hardcoded and inconsistent across scripts:

| Constant | Script 09 | Script 11 | Script 13 |
|---|---|---|---|
| Age threshold for "old" | 60 | 65+ (description) | 50 |
| Velocity threshold (I-VT) | — | — | hardcoded in 07 |
| Correlation cutoff (>0.85) | — | — | repeated in 4 scripts |
| Base data path | different in each script | | |

**Fix:** Create `gaze_analysis/config.py`:

```python
from pathlib import Path

ROOT = Path(__file__).parent

# Data paths
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"

# I-VT gaze algorithm
VELOCITY_THRESHOLD_CM_S    = 20.0
MIN_FIXATION_DURATION_SEC  = 0.100

# Age classification
AGE_THRESHOLD_OLD = 60  # unified threshold (was 50/60/65 across scripts)

# Feature filtering
MAX_INTER_FEATURE_CORR   = 0.85
MIN_FEATURE_TARGET_CORR  = 0.25

# Sync
SYNC_TOLERANCE_MS = 34
TARGET_HZ         = 30
```

Every script then does `from config import AGE_THRESHOLD_OLD` instead of hardcoding a number.

---

### 3. Extract Duplicated Feature-Filtering Logic into a Shared Utility (Priority: Medium)

**Problem:** The same two-stage feature filtering (drop inter-feature correlation > 0.85, then drop feature-target correlation < 0.25) is copy-pasted verbatim in four scripts: `10c_random_forest.py`, `11_train_model.py`, `11b_imbalance_data_training.py`, and `12_t-SNE.py`. Any change to the thresholds or logic must be applied in four places and is prone to drift.

**Fix:** Create `gaze_analysis/utils/feature_selection.py`:

```python
import pandas as pd
import numpy as np
from config import MAX_INTER_FEATURE_CORR, MIN_FEATURE_TARGET_CORR

def filter_features(df: pd.DataFrame, feature_cols: list, target_col: str) -> list:
    """Return feature columns surviving correlation-based filtering."""
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_inter = [c for c in upper.columns if any(upper[c] > MAX_INTER_FEATURE_CORR)]

    surviving = [c for c in feature_cols if c not in drop_inter]
    target_corr = df[surviving].corrwith(df[target_col]).abs()
    drop_low = target_corr[target_corr < MIN_FEATURE_TARGET_CORR].index.tolist()

    return [c for c in surviving if c not in drop_low]
```

Replace four copy-paste blocks with `from utils.feature_selection import filter_features`.

---

### 4. Parallelize Per-Participant Feature Extraction (Priority: Medium)

**Problem:** Scripts 05, 06, 07, and 08 iterate sequentially over participants with a `for participant in participants:` loop. With 50+ participants × 6 games each, this is the bottleneck of the pipeline. The computation for each participant is fully independent.

**Fix:** Wrap each script's per-participant work in a function and use `multiprocessing.Pool`:

```python
from multiprocessing import Pool, cpu_count
from pathlib import Path

def process_participant(participant_dir: Path) -> None:
    # existing per-participant logic moved here
    ...

if __name__ == "__main__":
    participant_dirs = sorted(Path(PROCESSED_DIR).iterdir())
    with Pool(processes=cpu_count() - 1) as pool:
        pool.map(process_participant, participant_dirs)
```

This change alone can reduce the wall-clock time of the feature extraction stage by 4–8× on a modern laptop with 8+ cores.

---

### 5. Add Pipeline Orchestration with a Makefile (Priority: Medium)

**Problem:** Running the full pipeline requires manually executing 13+ scripts in the correct order. There is no way to know which outputs are stale, no single command to reproduce results from scratch, and no record of which scripts were actually run.

**Fix:** Add a `Makefile` at the repository root:

```makefile
PYTHON = python
SCRIPTS = gaze_analysis/scripts

.PHONY: all sync features merge explore model clean

all: model

sync: $(SCRIPTS)/04_sync_and_move.py
	$(PYTHON) $<

features: sync
	$(PYTHON) $(SCRIPTS)/05_tapping_features.py
	$(PYTHON) $(SCRIPTS)/06_dragging_features.py
	$(PYTHON) $(SCRIPTS)/07_gaze_features.py
	$(PYTHON) $(SCRIPTS)/08_sync_features.py

merge: features
	$(PYTHON) $(SCRIPTS)/09_merge_matrices.py

explore: merge
	$(PYTHON) $(SCRIPTS)/10_explore_data.py
	$(PYTHON) $(SCRIPTS)/10c_random_forest.py

model: explore
	$(PYTHON) $(SCRIPTS)/11_train_model.py
	$(PYTHON) $(SCRIPTS)/11c_confusion_matrix.py
	$(PYTHON) $(SCRIPTS)/12_t-SNE.py

clean:
	rm -f gaze_analysis/data/processed/*.csv
	rm -f gaze_analysis/results/*.png gaze_analysis/results/*.csv
```

Running `make model` re-executes only what is needed; `make clean && make model` gives a fully reproducible run from synced timeseries.

---

## Summary Table

| # | Improvement | Focus Area | Effort |
|---|---|---|---|
| 1 | Add `requirements.txt` + `environment.yml` | Reproducibility | Low (1 hour) |
| 2 | Create `config.py`, fix age threshold inconsistency | Modularity, Correctness | Low–Medium |
| 3 | Extract feature-filtering into `utils/feature_selection.py` | Modularity | Low |
| 4 | Parallelize Scripts 05–08 with `multiprocessing.Pool` | Performance | Medium |
| 5 | Add `Makefile` for pipeline orchestration | Reproducibility, Usability | Medium |
