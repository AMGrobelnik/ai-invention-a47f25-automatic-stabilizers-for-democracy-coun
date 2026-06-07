"""Step 4: Triple dissociation via polynomial controls and CEM."""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def _build_lp_h1_data(panel: pd.DataFrame, country_df: pd.DataFrame,
                       h: int = 1) -> pd.DataFrame:
    """Build horizon h=1 LP dataset for dissociation tests."""
    panel = panel.merge(
        country_df[["iso3c", "structural_wedge", "cyclical_resp"]],
        on="iso3c", how="left"
    )
    rows = []
    for iso, grp in panel.groupby("iso3c"):
        grp = grp.set_index("year").sort_index()
        for yr in grp.index:
            yr_lag = yr - 1
            yr_fwd = yr + h
            if yr_lag not in grp.index or yr_fwd not in grp.index:
                continue
            y_lag = grp.loc[yr_lag, "v2x_libdem"] if "v2x_libdem" in grp.columns else np.nan
            y_fwd = grp.loc[yr_fwd, "v2x_libdem"] if "v2x_libdem" in grp.columns else np.nan
            if pd.isna(y_lag) or pd.isna(y_fwd):
                continue
            shock = grp.loc[yr, "gdp_innovation"] if "gdp_innovation" in grp.columns else np.nan
            if pd.isna(shock):
                continue
            row = {
                "iso3c": iso,
                "year": yr,
                "outcome": float(y_fwd - y_lag),
                "shock": float(shock),
                "structural_wedge": float(grp["structural_wedge"].iloc[0]),
                "cyclical_resp": float(grp["cyclical_resp"].iloc[0]),
                "net_gini_lagged": float(grp.loc[yr_lag, "gini_net"]) if "gini_net" in grp.columns else np.nan,
                "log_gdppc": float(grp.loc[yr_lag, "log_gdppc"]) if "log_gdppc" in grp.columns else np.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows).dropna()


def polynomial_dissociation(panel: pd.DataFrame, country_df: pd.DataFrame) -> Dict:
    """
    Add quadratic controls for net_gini and structural_wedge.
    Test whether cyclical_resp interaction remains significant.
    """
    logger.info("Running polynomial dissociation test...")
    df = _build_lp_h1_data(panel, country_df, h=1)
    if len(df) < 30:
        logger.warning("Insufficient data for polynomial test")
        return {"polynomial_delta_h_significant": None}

    # Entity and time demeaning
    for col in ["outcome", "shock", "net_gini_lagged", "log_gdppc"]:
        if col in df.columns:
            df[col] = df[col] - df.groupby("iso3c")[col].transform("mean")

    # Interactions
    df["shock_x_cyclical"] = df["shock"] * df["cyclical_resp"]
    df["shock_x_structural"] = df["shock"] * df["structural_wedge"]
    df["shock_x_netgini"] = df["shock"] * df.get("net_gini_lagged", 40.0)

    # Polynomial controls
    df["structural_sq"] = df["structural_wedge"] ** 2
    df["netgini_sq"] = df["net_gini_lagged"] ** 2 if "net_gini_lagged" in df.columns else 0
    df["shock_x_struct_sq"] = df["shock"] * df["structural_sq"]
    df["shock_x_netgini_sq"] = df["shock"] * df["netgini_sq"]

    year_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True)
    feat_cols = ["shock", "shock_x_cyclical", "shock_x_structural", "shock_x_netgini",
                 "shock_x_struct_sq", "shock_x_netgini_sq",
                 "net_gini_lagged", "log_gdppc"]
    feat_cols = [c for c in feat_cols if c in df.columns and df[c].std() > 1e-8]
    X_df = pd.concat([df[feat_cols], year_dummies], axis=1).fillna(0)
    y = df["outcome"].values

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = OLS(y, X_df.values).fit(cov_type="HC3")
        idx = X_df.columns.tolist().index("shock_x_cyclical")
        tstat = res.params[idx] / res.bse[idx]
        pval = float(2 * (1 - sp_stats.t.cdf(abs(tstat), df=len(df) - X_df.shape[1])))
        significant = pval < 0.10
        logger.info(f"Polynomial test: delta={res.params[idx]:.4f}, p={pval:.4f}, sig={significant}")
        return {"polynomial_delta_h_significant": bool(significant), "polynomial_pvalue": float(pval),
                "polynomial_delta_coef": float(res.params[idx])}
    except Exception as e:
        logger.warning(f"Polynomial test failed: {e}")
        return {"polynomial_delta_h_significant": None}


def cem_dissociation(panel: pd.DataFrame, country_df: pd.DataFrame) -> Dict:
    """
    Coarsened Exact Matching on (structural_wedge_bin, net_gini_bin).
    Run LP on matched sample.
    """
    logger.info("Running CEM dissociation test...")
    df = _build_lp_h1_data(panel, country_df, h=1)
    if len(df) < 30:
        logger.warning("Insufficient data for CEM")
        return {"cem_n_matched": 0, "cem_delta_h_significant": None}

    # Bin structural_wedge and net_gini into quartiles
    df["struct_bin"] = pd.qcut(df["structural_wedge"], q=4, labels=False, duplicates="drop")
    if "net_gini_lagged" in df.columns and df["net_gini_lagged"].std() > 0:
        df["netgini_bin"] = pd.qcut(df["net_gini_lagged"], q=4, labels=False, duplicates="drop")
    else:
        df["netgini_bin"] = 0

    df["cem_cell"] = df["struct_bin"].astype(str) + "_" + df["netgini_bin"].astype(str)

    # Keep only cells with > 1 unique cyclical_resp tercile
    cyc_q = df["cyclical_resp"].quantile([0.33, 0.67])
    df["cyc_tercile"] = pd.cut(df["cyclical_resp"],
                                bins=[-np.inf, cyc_q.iloc[0], cyc_q.iloc[1], np.inf],
                                labels=[0, 1, 2])

    cell_tercile_counts = df.groupby("cem_cell")["cyc_tercile"].nunique()
    valid_cells = cell_tercile_counts[cell_tercile_counts > 1].index
    df_matched = df[df["cem_cell"].isin(valid_cells)].copy()

    n_matched = len(df_matched)
    n_dropped = len(df) - n_matched
    logger.info(f"CEM: {n_matched} matched, {n_dropped} dropped")

    if n_matched < 20:
        logger.warning("CEM: too few matched units, trying tertile binning")
        # Try coarser bins
        df["struct_bin"] = pd.qcut(df["structural_wedge"], q=3, labels=False, duplicates="drop")
        if "net_gini_lagged" in df.columns and df["net_gini_lagged"].std() > 0:
            df["netgini_bin"] = pd.qcut(df["net_gini_lagged"], q=3, labels=False, duplicates="drop")
        df["cem_cell"] = df["struct_bin"].astype(str) + "_" + df["netgini_bin"].astype(str)
        cell_tercile_counts = df.groupby("cem_cell")["cyc_tercile"].nunique()
        valid_cells = cell_tercile_counts[cell_tercile_counts > 1].index
        df_matched = df[df["cem_cell"].isin(valid_cells)].copy()
        n_matched = len(df_matched)

    if n_matched < 15:
        return {"cem_n_matched": n_matched, "cem_delta_h_significant": None}

    # Run LP on matched sample
    for col in ["outcome", "shock"]:
        df_matched[col] = df_matched[col] - df_matched.groupby("iso3c")[col].transform("mean")

    df_matched["shock_x_cyclical"] = df_matched["shock"] * df_matched["cyclical_resp"]
    cell_dummies = pd.get_dummies(df_matched["cem_cell"], prefix="cell", drop_first=True)
    feat_cols = ["shock", "shock_x_cyclical"]
    X_df = pd.concat([df_matched[feat_cols], cell_dummies], axis=1).fillna(0)
    y = df_matched["outcome"].values

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = OLS(y, X_df.values).fit(cov_type="HC3")
        idx = X_df.columns.tolist().index("shock_x_cyclical")
        tstat = res.params[idx] / res.bse[idx]
        pval = float(2 * (1 - sp_stats.t.cdf(abs(tstat), df=len(df_matched) - X_df.shape[1])))
        significant = pval < 0.10
        logger.info(f"CEM: delta={res.params[idx]:.4f}, p={pval:.4f}, sig={significant}")
        return {
            "cem_n_matched": n_matched,
            "cem_delta_h_significant": bool(significant),
            "cem_pvalue": float(pval),
            "cem_delta_coef": float(res.params[idx]),
        }
    except Exception as e:
        logger.warning(f"CEM regression failed: {e}")
        return {"cem_n_matched": n_matched, "cem_delta_h_significant": None}


def run_triple_dissociation(panel: pd.DataFrame, country_df: pd.DataFrame) -> Dict:
    poly = polynomial_dissociation(panel, country_df)
    cem = cem_dissociation(panel, country_df)
    return {**poly, **cem}
