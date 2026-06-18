"""
Script 18 – Per-Patient OT Report Generator
============================================
Builds a clinician-facing PDF report per patient: a plain-language summary of
their gaze + touch performance across all 6 games, with each metric placed
against same-age-group peers (percentile bands), grouped into clinically
meaningful domains (motor planning, motor control, precision, visual
attention, visual search, eye-hand coordination, engagement), plus a
SHAP-derived modality summary (Touch / Gaze / Sync) translated into plain
language.

Renders an HTML/Jinja2 template to PDF via WeasyPrint.

Usage:
    python 18_generate_patient_report.py --patient janan
    python 18_generate_patient_report.py --all

Outputs → gaze_analysis/results/patient_reports/<Patient_ID>_report.pdf
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from scipy.stats import percentileofscore
from weasyprint import HTML

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import ALL_GAMES, PROCESSED_DIR, RESULTS_DIR
from utils.report_content_less_features import (
    DOMAIN_ORDER,
    DOMAINS,
    FEATURES,
    GAME_DESCRIPTIONS,
    modality_summary,
    percentile_band,
)

ABLATION_DIR = RESULTS_DIR / "per_patient_ablation"
TEMPLATES_DIR = SCRIPT_DIR.parent / "templates"
OUTPUT_DIR = ABLATION_DIR.parent / "patient_reports"

# Columns present in MASTER_*.csv that are identifiers/targets, not metrics
NON_METRIC_COLS = {"Patient_ID", "Game", "Age", "Target_Age_Old", "Target_Visual_Impaired"}

N_HIGHLIGHTS = 4


# ── Data loading ─────────────────────────────────────────────────────────────

def load_master_tables():
    """
    Load all 6 MASTER_<Game>.csv files. Returns (master_df, game_columns):
    - master_df: union of all games concatenated (used for peer percentiles)
    - game_columns: {game: set(native columns)} — games have different
      feature sets (e.g. only dragging games have Spatial_Jerk_Magnitude),
      and pd.concat fills the non-native ones with NaN. We need the native
      set to avoid showing "not measured" for features a game never collects.
    """
    frames = []
    game_columns = {}
    for game in ALL_GAMES:
        df = pd.read_csv(PROCESSED_DIR / f"MASTER_{game}.csv")
        game_columns[game] = set(df.columns)
        frames.append(df)
    return pd.concat(frames, ignore_index=True), game_columns


def load_support_tables():
    modality = pd.read_csv(ABLATION_DIR / "per_patient_modality_contribution.csv")
    accuracy = pd.read_csv(ABLATION_DIR / "per_patient_accuracy_scores.csv")
    return modality, accuracy


# ── Percentile computation (within same-age-group peers) ────────────────────

def compute_peer_percentile(master_df, game, feature, patient_value, age_group):
    """
    Percentile (0-100) of `patient_value` for `feature` in `game`, computed
    only against peers in the same `age_group` (Target_Age_Old == age_group).
    Returns None if the feature/value isn't available or there's no peer data.
    """
    if pd.isna(patient_value):
        return None
    peer_rows = master_df[
        (master_df["Game"] == game) & (master_df["Target_Age_Old"] == age_group)
    ]
    if feature not in peer_rows.columns:
        return None
    peer_values = peer_rows[feature].dropna()
    if peer_values.empty:
        return None
    return float(percentileofscore(peer_values, patient_value, kind="mean"))


# ── Per-patient context assembly ─────────────────────────────────────────────

def build_game_context(master_df, game, patient_row, age_group, native_columns):
    """Build the {description, domains:[{label, blurb, metrics:[...]}]} block for one game."""
    feature_cols = [c for c in patient_row.index if c in FEATURES and c in native_columns]

    domain_metrics = {d: [] for d in DOMAIN_ORDER}
    for feature in feature_cols:
        meta = FEATURES[feature]
        raw_value = patient_row[feature]
        value = None if pd.isna(raw_value) else round(float(raw_value), 2)
        pct = compute_peer_percentile(master_df, game, feature, raw_value, age_group)
        domain_metrics[meta["domain"]].append({
            "label": meta["label"],
            "unit": meta["unit"],
            "note": meta["note"],
            "value": value,
            "percentile": pct,
            "band": percentile_band(pct),
        })

    domains = []
    for domain_key in DOMAIN_ORDER:
        metrics = domain_metrics[domain_key]
        if not metrics:
            continue
        domains.append({
            "label": DOMAINS[domain_key]["label"],
            "blurb": DOMAINS[domain_key]["blurb"],
            "metrics": metrics,
        })

    return {
        "name": game,
        "description": GAME_DESCRIPTIONS[game],
        "domains": domains,
    }


def collect_highlights(games_ctx, n=N_HIGHLIGHTS):
    """
    Pick the metrics whose peer-percentile deviates most from the median (50),
    in either direction, as "notable patterns" for the executive summary.
    """
    candidates = []
    for game in games_ctx:
        for domain in game["domains"]:
            for m in domain["metrics"]:
                if m["percentile"] is None:
                    continue
                deviation = abs(m["percentile"] - 50)
                if deviation < 25:  # only surface genuinely atypical results
                    continue
                candidates.append({
                    "game": game["name"],
                    "label": m["label"],
                    "value": m["value"],
                    "unit": (" " + m["unit"]) if m["unit"] else "",
                    "band": m["band"],
                    "deviation": deviation,
                })
    candidates.sort(key=lambda c: c["deviation"], reverse=True)
    return candidates[:n]


def build_patient_context(patient_id, master_df, game_columns, modality_df, accuracy_df):
    patient_master = master_df[master_df["Patient_ID"] == patient_id]
    if patient_master.empty:
        raise ValueError(f"No data found for patient '{patient_id}'")

    first_row = patient_master.iloc[0]
    age = int(first_row["Age"])
    age_group = int(first_row["Target_Age_Old"])

    acc_row = accuracy_df[accuracy_df["Patient_ID"] == patient_id]
    age_group_label = (
        acc_row.iloc[0]["Age_Group_Label"] if not acc_row.empty
        else ("Old (≥60)" if age_group == 1 else "Young (<60)")
    )
    peer_group_size = int((master_df.drop_duplicates("Patient_ID")["Target_Age_Old"] == age_group).sum())
    cohort_size = master_df["Patient_ID"].nunique()

    games_ctx = []
    for game in ALL_GAMES:
        rows = patient_master[patient_master["Game"] == game]
        if rows.empty:
            continue
        games_ctx.append(build_game_context(master_df, game, rows.iloc[0], age_group, game_columns[game]))

    mod_row = modality_df[modality_df["Patient_ID"] == patient_id]
    if not mod_row.empty:
        mod_row = mod_row.iloc[0]
        mod_summary = modality_summary(
            mod_row["Touch_pct"], mod_row["Gaze_pct"], mod_row["Sync_pct"], mod_row["dominant_modality"]
        )
    else:
        mod_summary = "Modality attribution data is not available for this patient."

    return {
        "patient_id": patient_id,
        "age": age,
        "age_group_label": age_group_label,
        "peer_group_size": peer_group_size,
        "cohort_size": cohort_size,
        "generated_date": date.today().isoformat(),
        "modality_summary": mod_summary,
        "games": games_ctx,
        "highlights": collect_highlights(games_ctx),
    }


# ── Rendering ────────────────────────────────────────────────────────────────

def render_pdf(context, output_path):
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("patient_report.html")
    html_str = template.render(**context)
    HTML(string=html_str, base_url=str(TEMPLATES_DIR)).write_pdf(str(output_path))


def generate_report(patient_id, master_df, game_columns, modality_df, accuracy_df):
    context = build_patient_context(patient_id, master_df, game_columns, modality_df, accuracy_df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{patient_id}_report.pdf"
    render_pdf(context, output_path)
    print(f"  ✓ {patient_id} → {output_path.relative_to(RESULTS_DIR.parent)}")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate per-patient OT PDF reports.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient", help="Patient_ID to generate a report for, e.g. 'janan'")
    group.add_argument("--all", action="store_true", help="Generate reports for all patients")
    args = parser.parse_args()

    master_df, game_columns = load_master_tables()
    modality_df, accuracy_df = load_support_tables()

    if args.all:
        patient_ids = sorted(master_df["Patient_ID"].unique())
        print(f"Generating {len(patient_ids)} patient reports → {OUTPUT_DIR}")
        for pid in patient_ids:
            generate_report(pid, master_df, game_columns, modality_df, accuracy_df)
    else:
        generate_report(args.patient, master_df, game_columns, modality_df, accuracy_df)


if __name__ == "__main__":
    main()
