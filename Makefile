PYTHON  := conda run -n tatoo python
SCRIPTS := gaze_analysis/scripts

.PHONY: all organize sync features merge explore model tsne clean help

# ── Default target ────────────────────────────────────────────────────────────
all: model tsne

# ── Step 1: organize raw files by participant ─────────────────────────────────
organize:
	$(PYTHON) $(SCRIPTS)/01_organize_new_data.py

# ── Step 2: synchronize gaze + touch into 30 Hz timeseries ───────────────────
sync: organize
	$(PYTHON) $(SCRIPTS)/04_sync_and_move.py

# ── Step 3: extract features (tapping, dragging, gaze, eye-hand sync) ─────────
features: sync
	$(PYTHON) $(SCRIPTS)/05_tapping_features.py
	$(PYTHON) $(SCRIPTS)/06_dragging_features.py
	$(PYTHON) $(SCRIPTS)/07_gaze_features.py
	$(PYTHON) $(SCRIPTS)/08_sync_features.py

# ── Step 4: merge feature matrices and attach demographics ───────────────────
merge: features
	$(PYTHON) $(SCRIPTS)/09_merge_matrices.py

# ── Step 5: exploratory data analysis and feature audit ──────────────────────
explore: merge
	$(PYTHON) $(SCRIPTS)/10_explore_data.py
	$(PYTHON) $(SCRIPTS)/10c_random_forest.py

# ── Step 6: train model and generate per-patient performance matrix ───────────
model: explore
	$(PYTHON) $(SCRIPTS)/11_train_model.py
	$(PYTHON) $(SCRIPTS)/11c_confusion_matrix.py

# ── Step 7: dimensionality reduction visualisation ────────────────────────────
tsne: merge
	$(PYTHON) $(SCRIPTS)/12_t-SNE.py

# ── Convenience: violin plots (independent of model training) ─────────────────
violins: merge
	$(PYTHON) $(SCRIPTS)/10b_explore_data.py

# ── Convenience: imbalanced-data model variant ───────────────────────────────
model-imbalanced: explore
	$(PYTHON) $(SCRIPTS)/11b_imbalance_data_training.py

# ── Clean generated outputs (keeps raw data and synced timeseries) ────────────
clean:
	rm -f gaze_analysis/data/processed/*_ML_Features.csv
	rm -f gaze_analysis/data/processed/*_Gaze_Features.csv
	rm -f gaze_analysis/data/processed/*_Sync_Features.csv
	rm -f gaze_analysis/data/processed/MASTER_*.csv
	rm -f gaze_analysis/results/*.png gaze_analysis/results/*.csv
	rm -rf gaze_analysis/results/correlation_heatmaps
	rm -rf gaze_analysis/results/tSNE

# ── Remove everything including synced timeseries (full re-run from raw) ──────
clean-all: clean
	rm -f gaze_analysis/data/processed/*_synced_timeseries.csv

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "── OLD APP PIPELINE ──────────────────────────────────────"
	@echo "  all              Run full pipeline (model + tsne)"
	@echo "  organize         Step 1 – route raw files into participant folders"
	@echo "  sync             Step 2 – synchronise gaze + touch at 30 Hz"
	@echo "  features         Step 3 – extract tapping/dragging/gaze/sync features"
	@echo "  merge            Step 4 – merge feature matrices + demographics"
	@echo "  explore          Step 5 – EDA, correlation heatmaps, RF audit"
	@echo "  model            Step 6 – XGBoost training + per-patient matrix"
	@echo "  tsne             Step 7 – t-SNE dimensionality reduction"
	@echo "  violins          Violin plots (after merge)"
	@echo "  model-imbalanced Imbalanced-data model variant (after explore)"
	@echo "  clean            Remove derived outputs, keep synced timeseries"
	@echo "  clean-all        Remove everything including synced timeseries"
	@echo ""
	@echo "── NEW APP PIPELINE (8 games, no demographics yet) ───────"
	@echo "  new-app-organize  Step 1 – route raw_new_app/ files into subfolders"
	@echo "  new-app-sync      Step 2 – sync NPZ at 30 Hz"
	@echo "  new-app-features  Step 3 – extract all features (incl. HoldIt)"
	@echo "  new-app-merge     Step 4 – merge into MASTER_{game}.csv (no labels)"
	@echo "  new-app-explore   Step 5 – EDA + violin plots"
	@echo "  new-app-all       Run steps 1-5 (full pipeline without model)"
	@echo "  new-app-clean     Remove new-app derived outputs"
	@echo "  (new-app-model and new-app-ensemble are commented out until"
	@echo "   demographics/labels are collected)"

# ═══════════════════════════════════════════════════════════════════════════════
# NEW APP PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
NEW := gaze_analysis/scripts_new_app

.PHONY: new-app-organize new-app-sync new-app-features new-app-merge \
        new-app-explore new-app-all new-app-clean

new-app-organize:
	$(PYTHON) $(NEW)/01_organize_new_data.py

new-app-sync: new-app-organize
	$(PYTHON) $(NEW)/04_sync_and_move.py

new-app-features: new-app-sync
	$(PYTHON) $(NEW)/05_tapping_features.py
	$(PYTHON) $(NEW)/06_dragging_features.py
	$(PYTHON) $(NEW)/07_gaze_features.py
	$(PYTHON) $(NEW)/08_sync_features.py
	$(PYTHON) $(NEW)/15_holdit_features.py

new-app-merge: new-app-features
	$(PYTHON) $(NEW)/09_merge_matrices.py

new-app-explore: new-app-merge
	$(PYTHON) $(NEW)/10_explore_data.py
	$(PYTHON) $(NEW)/10b_explore_data.py

new-app-all: new-app-explore
	@echo "New-app pipeline complete (EDA only)."
	@echo "Activate new-app-model after demographics are collected."

# ── Activate these targets when demographics/labels become available ──────────
# new-app-model: new-app-merge
#	$(PYTHON) $(NEW)/11_train_model.py
#
# new-app-ensemble: new-app-model
#	$(PYTHON) $(NEW)/17_weighted_ensemble.py

new-app-transfer: new-app-merge
	$(PYTHON) $(NEW)/18_apply_old_model.py

new-app-clean:
	rm -f gaze_analysis/data/processed_new_app/*_ML_Features.csv
	rm -f gaze_analysis/data/processed_new_app/*_Gaze_Features.csv
	rm -f gaze_analysis/data/processed_new_app/*_Sync_Features.csv
	rm -f gaze_analysis/data/processed_new_app/MASTER_*.csv
	rm -rf gaze_analysis/results/new_app
