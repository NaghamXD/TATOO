"""
Shared feature-filtering logic used by scripts 10c, 11, 11b, and 12.
Previously copy-pasted in each script — centralised here so threshold
changes propagate everywhere automatically.
"""
import numpy as np
import pandas as pd

from config import (
    MAX_INTER_FEATURE_CORR,
    MIN_FEATURE_TARGET_CORR,
    MISSING_DROP_THRESHOLD,
)


def filter_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    verbose: bool = False,
) -> list[str]:
    """Return the subset of feature_cols that survive three-stage filtering.

    Stage 1 – Drop features missing in more than MISSING_DROP_THRESHOLD of rows.
    Stage 2 – Drop one column from every inter-feature pair whose absolute
               Pearson correlation exceeds MAX_INTER_FEATURE_CORR (keeps the
               column that appears first in feature_cols).
    Stage 3 – Drop features whose absolute correlation with target_col is
               below MIN_FEATURE_TARGET_CORR.
    """
    candidates = [c for c in feature_cols if c in df.columns]

    # Stage 1: missing data
    missing_rate = df[candidates].isnull().mean()
    drop_missing = missing_rate[missing_rate > MISSING_DROP_THRESHOLD].index.tolist()
    candidates = [c for c in candidates if c not in drop_missing]
    if verbose and drop_missing:
        print(f"[filter] dropped {len(drop_missing)} high-missing features: {drop_missing}")

    # Stage 2: inter-feature correlation
    corr_matrix = df[candidates].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )
    drop_inter = [c for c in upper.columns if upper[c].max() > MAX_INTER_FEATURE_CORR]
    candidates = [c for c in candidates if c not in drop_inter]
    if verbose and drop_inter:
        print(f"[filter] dropped {len(drop_inter)} high-inter-correlation features: {drop_inter}")

    # Stage 3: feature–target correlation
    if target_col not in df.columns:
        return candidates
    target_corr = df[candidates].corrwith(df[target_col]).abs()
    drop_low = target_corr[target_corr < MIN_FEATURE_TARGET_CORR].index.tolist()
    candidates = [c for c in candidates if c not in drop_low]
    if verbose and drop_low:
        print(f"[filter] dropped {len(drop_low)} low-target-correlation features: {drop_low}")

    return candidates


def split_features_by_status(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[list[str], list[str], list[str]]:
    """Return three lists: (surviving, dropped_inter_corr, dropped_low_relevance).

    Useful for audit heatmaps (script 10c) where all three groups need labels.
    """
    candidates = [c for c in feature_cols if c in df.columns]

    missing_rate = df[candidates].isnull().mean()
    drop_missing = missing_rate[missing_rate > MISSING_DROP_THRESHOLD].index.tolist()
    candidates = [c for c in candidates if c not in drop_missing]

    corr_matrix = df[candidates].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    drop_inter = [c for c in upper.columns if upper[c].max() > MAX_INTER_FEATURE_CORR]
    after_inter = [c for c in candidates if c not in drop_inter]

    target_corr = df[after_inter].corrwith(df[target_col]).abs()
    drop_low = target_corr[target_corr < MIN_FEATURE_TARGET_CORR].index.tolist()
    surviving = [c for c in after_inter if c not in drop_low]

    return surviving, drop_inter, drop_low
