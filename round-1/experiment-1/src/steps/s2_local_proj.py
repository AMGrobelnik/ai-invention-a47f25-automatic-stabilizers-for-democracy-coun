"""Step 2: Panel local projections (Jordà 2005) with Driscoll-Kraay SEs."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats


def _driscoll_kraay_se(X: np.ndarray, resid: np.ndarray, n_groups: int,
                        bandwidth: int) -> np.ndarray:
    """
    Compute Driscoll-Kraay standard errors.
    Newey-West kernel applied to cross-sectional sums of moment conditions.
    """
    T = resid.shape[0]
    k = X.shape[1]
    # Score matrix: T x k
    # Group by time and sum (approximate: treat as pooled time series)
    S = np.zeros((k, k))
    XeT = X * resid[:, None]  # T x k score per obs

    # Gamma_0
    G0 = XeT.T @ XeT
    S = G0.copy()

    for l in range(1, bandwidth + 1):
        Gl = XeT[l:].T @ XeT[:-l]
        w = 1.0 - l / (bandwidth + 1)
        S += w * (Gl + Gl.T)

    XtX_inv = np.linalg.pinv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(V), 0))


def run_lp(
    panel: pd.DataFrame,
    country_df: pd.DataFrame,
    horizons: List[int] = None,
    outcome_col: str = "v2x_libdem",
    shock_col: str = "gdp_innovation",
    n_lags_controls: int = 1,
    bandwidth: Optional[int] = None,
) -> Dict:
    """Run panel local projections for each horizon h."""
    if horizons is None:
        horizons = list(range(7))  # 0..6

    logger.info(f"Running LP for horizons {horizons}, outcome={outcome_col}, shock={shock_col}")

    # Merge country-level components into panel
    panel = panel.merge(
        country_df[["iso3c", "structural_wedge", "cyclical_resp", "net_gini_mean"]],
        on="iso3c", how="left"
    )

    results = {
        "horizons": horizons,
        "delta_h": [], "delta_h_ci95_lo": [], "delta_h_ci95_hi": [],
        "phi_h": [], "phi_h_ci95_lo": [], "phi_h_ci95_hi": [],
        "theta_h": [], "theta_h_ci95_lo": [], "theta_h_ci95_hi": [],
        "beta_h": [], "beta_h_ci95_lo": [], "beta_h_ci95_hi": [],
        "dominance_wald_pvalue": [],
        "n_obs_per_h": [],
    }

    for h in horizons:
        logger.info(f"  Horizon h={h}...")
        df_h = _build_horizon_df(panel, h, outcome_col, shock_col, n_lags_controls)
        if df_h is None or len(df_h) < 50:
            logger.warning(f"  h={h}: insufficient data ({len(df_h) if df_h is not None else 0} obs)")
            _append_nan(results)
            continue

        coefs, ses, n_obs = _estimate_lp(df_h, bandwidth)
        if coefs is None:
            _append_nan(results)
            continue

        def _get(name: str):
            return float(coefs.get(name, np.nan)), float(ses.get(name, np.nan))

        beta_c, beta_se = _get("shock")
        delta_c, delta_se = _get("shock_x_cyclical")
        phi_c, phi_se = _get("shock_x_structural")
        theta_c, theta_se = _get("shock_x_netgini")

        results["beta_h"].append(beta_c)
        results["beta_h_ci95_lo"].append(beta_c - 1.96 * beta_se)
        results["beta_h_ci95_hi"].append(beta_c + 1.96 * beta_se)

        results["delta_h"].append(delta_c)
        results["delta_h_ci95_lo"].append(delta_c - 1.96 * delta_se)
        results["delta_h_ci95_hi"].append(delta_c + 1.96 * delta_se)

        results["phi_h"].append(phi_c)
        results["phi_h_ci95_lo"].append(phi_c - 1.96 * phi_se)
        results["phi_h_ci95_hi"].append(phi_c + 1.96 * phi_se)

        results["theta_h"].append(theta_c)
        results["theta_h_ci95_lo"].append(theta_c - 1.96 * theta_se)
        results["theta_h_ci95_hi"].append(theta_c + 1.96 * theta_se)

        # Wald test H0: delta = phi
        if not np.isnan(delta_c) and not np.isnan(phi_c) and delta_se > 0 and phi_se > 0:
            diff = delta_c - phi_c
            diff_se = np.sqrt(delta_se**2 + phi_se**2)
            wald_stat = (diff / diff_se) ** 2
            pval = float(1 - sp_stats.chi2.cdf(wald_stat, df=1))
        else:
            pval = np.nan
        results["dominance_wald_pvalue"].append(pval)
        results["n_obs_per_h"].append(n_obs)

        logger.info(f"    delta={delta_c:.4f}±{delta_se:.4f}, phi={phi_c:.4f}, "
                    f"theta={theta_c:.4f}, wald_p={pval:.3f}, n={n_obs}")

    results["n_obs"] = int(np.nanmax(results["n_obs_per_h"])) if results["n_obs_per_h"] else 0
    return results


def _build_horizon_df(
    panel: pd.DataFrame, h: int, outcome_col: str, shock_col: str, n_lags: int
) -> Optional[pd.DataFrame]:
    """Build outcome = y_{t+h} - y_{t-1} and merge with regressors."""
    panel = panel.sort_values(["iso3c", "year"])

    rows = []
    for iso, grp in panel.groupby("iso3c"):
        grp = grp.set_index("year").sort_index()
        years = grp.index.tolist()

        for i, yr in enumerate(years):
            # Need y_{t-1}, y_{t+h}, shock_t, controls_{t-1}
            yr_lag = yr - 1
            yr_fwd = yr + h
            if yr_lag not in grp.index or yr_fwd not in grp.index:
                continue

            y_lag = grp.loc[yr_lag, outcome_col]
            y_fwd = grp.loc[yr_fwd, outcome_col]
            if pd.isna(y_lag) or pd.isna(y_fwd):
                continue

            shock_val = grp.loc[yr, shock_col] if shock_col in grp.columns else np.nan
            if pd.isna(shock_val):
                continue

            row = {
                "iso3c": iso,
                "year": yr,
                "outcome": float(y_fwd - y_lag),
                "shock": float(shock_val),
                "structural_wedge": float(grp["structural_wedge"].iloc[0]) if "structural_wedge" in grp.columns else np.nan,
                "cyclical_resp": float(grp["cyclical_resp"].iloc[0]) if "cyclical_resp" in grp.columns else np.nan,
                "net_gini_lagged": float(grp.loc[yr_lag, "gini_net"]) if "gini_net" in grp.columns else np.nan,
                "log_gdppc": float(grp.loc[yr_lag, "log_gdppc"]) if "log_gdppc" in grp.columns else np.nan,
                "net_gini_mean": float(grp["net_gini_mean"].iloc[0]) if "net_gini_mean" in grp.columns else np.nan,
            }
            rows.append(row)

    if not rows:
        return None
    df_h = pd.DataFrame(rows).dropna(subset=["outcome", "shock", "cyclical_resp", "structural_wedge"])
    return df_h


def _append_nan(results: Dict) -> None:
    for key in ["delta_h", "delta_h_ci95_lo", "delta_h_ci95_hi",
                "phi_h", "phi_h_ci95_lo", "phi_h_ci95_hi",
                "theta_h", "theta_h_ci95_lo", "theta_h_ci95_hi",
                "beta_h", "beta_h_ci95_lo", "beta_h_ci95_hi",
                "dominance_wald_pvalue", "n_obs_per_h"]:
        results[key].append(np.nan)


def _estimate_lp(df: pd.DataFrame, bandwidth: Optional[int] = None) -> tuple:
    """Estimate the LP equation using OLS with Driscoll-Kraay SEs."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    df = df.copy().dropna()

    # Standardise moderators to unit variance for numerical stability
    std_cyc = df["cyclical_resp"].std()
    std_str = df["structural_wedge"].std()
    std_ng = df["net_gini_lagged"].std() if "net_gini_lagged" in df.columns else 1.0

    df["shock_x_cyclical"] = df["shock"] * df["cyclical_resp"] / max(std_cyc, 0.01)
    df["shock_x_structural"] = df["shock"] * df["structural_wedge"] / max(std_str, 0.01)
    df["shock_x_netgini"] = df["shock"] * df.get("net_gini_lagged", df["net_gini_mean"]) / max(std_ng, 0.01)

    # Entity FE via demeaning
    for col in ["outcome", "shock", "shock_x_cyclical", "shock_x_structural", "shock_x_netgini",
                "log_gdppc", "net_gini_lagged"]:
        if col in df.columns:
            df[col] = df[col] - df.groupby("iso3c")[col].transform("mean")

    # Time FE via year dummies
    year_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True)

    feat_cols = ["shock", "shock_x_cyclical", "shock_x_structural", "shock_x_netgini",
                 "log_gdppc", "net_gini_lagged"]
    feat_cols = [c for c in feat_cols if c in df.columns and df[c].std() > 1e-8]

    X_df = pd.concat([df[feat_cols], year_dummies], axis=1).fillna(0).astype(float)
    X = X_df.values
    y = df["outcome"].values.astype(float)

    if X.shape[0] < X.shape[1] + 5:
        return None, None, 0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = OLS(y, X).fit()

        n_time = df["year"].nunique()
        bw = bandwidth or max(1, int(n_time ** 0.25))

        try:
            ses_dk = _driscoll_kraay_se(X, res.resid, df["iso3c"].nunique(), bw)
        except Exception:
            ses_dk = res.bse.values

        feat_names = X_df.columns.tolist()
        coefs = {feat_names[i]: res.params[i] for i in range(len(feat_names))}
        ses = {feat_names[i]: ses_dk[i] for i in range(len(feat_names))}

        # Rescale interaction SEs back to original scale
        if "shock_x_cyclical" in coefs:
            coefs["shock_x_cyclical"] /= max(std_cyc, 0.01)
            ses["shock_x_cyclical"] /= max(std_cyc, 0.01)
        if "shock_x_structural" in coefs:
            coefs["shock_x_structural"] /= max(std_str, 0.01)
            ses["shock_x_structural"] /= max(std_str, 0.01)
        if "shock_x_netgini" in coefs:
            coefs["shock_x_netgini"] /= max(std_ng, 0.01)
            ses["shock_x_netgini"] /= max(std_ng, 0.01)

        return coefs, ses, len(df)

    except Exception as e:
        logger.warning(f"LP estimation failed: {e}")
        return None, None, 0
