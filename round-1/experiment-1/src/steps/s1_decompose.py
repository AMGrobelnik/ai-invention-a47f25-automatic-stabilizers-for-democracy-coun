"""Step 1: Wedge decomposition into structural and cyclical components."""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def decompose_wedge(panel: pd.DataFrame, n_draws: int = 100) -> Dict:
    """
    Decompose redistribution wedge into structural (time-mean) and
    cyclical (GDP-sensitivity) components, propagated over n_draws
    synthetic MI draws using Rubin's rules.
    """
    logger.info(f"Decomposing wedge over {n_draws} draws...")
    panel = panel.copy()

    # Within-country SD diagnostic
    within_sds = panel.groupby("iso3c")["abs_red"].std(ddof=1).dropna()
    mean_within_sd = within_sds.mean()
    logger.info(f"Mean within-country SD(abs_red): {mean_within_sd:.3f}")

    testable = mean_within_sd >= 1.0
    if not testable:
        logger.warning("Within-country SD < 1 Gini point — cyclical slope may be unestimable")

    rng = np.random.default_rng(42)
    iso_list = panel["iso3c"].unique()

    # Store per-draw results
    draw_struct = []  # shape: (n_draws, n_countries)
    draw_cycl = []

    for draw_k in range(n_draws):
        # Synthesise draw: abs_red_k = mean + se * N(0,1)
        abs_red_k = panel["abs_red"] + panel["abs_red_se"] * rng.standard_normal(len(panel))
        panel_k = panel.copy()
        panel_k["abs_red_k"] = abs_red_k

        struct_k = {}
        cycl_k = {}

        for iso in iso_list:
            grp = panel_k[panel_k["iso3c"] == iso].dropna(subset=["abs_red_k", "gdp_growth"])
            if len(grp) < 5:
                continue
            struct_k[iso] = grp["abs_red_k"].mean()

            # OLS: abs_red = alpha + beta*gdp_growth + gamma*year
            grp = grp.copy()
            grp["t"] = grp["year"] - grp["year"].mean()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    X = add_constant(grp[["gdp_growth", "t"]])
                    res = OLS(grp["abs_red_k"], X).fit()
                    beta = res.params.get("gdp_growth", 0.0)
                    cycl_k[iso] = -float(beta)  # flip: positive = counter-cyclical
                except Exception:
                    cycl_k[iso] = 0.0

        draw_struct.append(struct_k)
        draw_cycl.append(cycl_k)

    # Pool via Rubin's rules
    all_iso = set()
    for d in draw_struct:
        all_iso.update(d.keys())
    all_iso = sorted(all_iso)

    struct_means = np.array([[d.get(iso, np.nan) for iso in all_iso] for d in draw_struct])
    cycl_means = np.array([[d.get(iso, np.nan) for iso in all_iso] for d in draw_cycl])

    structural_wedge = np.nanmean(struct_means, axis=0)
    cyclical_resp = np.nanmean(cycl_means, axis=0)

    # Between-draw variance for Rubin's formula
    def rubins_var(draws: np.ndarray) -> np.ndarray:
        m = np.nanmean(draws, axis=0)
        between = np.nanvar(draws, axis=0, ddof=1)
        return between * (1 + 1 / draws.shape[0])

    cyclical_resp_var = rubins_var(cycl_means)
    cyclical_resp_se = np.sqrt(np.maximum(cyclical_resp_var, 0))

    # Build country-level dataframe
    country_df = pd.DataFrame({
        "iso3c": all_iso,
        "structural_wedge": structural_wedge,
        "cyclical_resp": cyclical_resp,
        "cyclical_resp_se": cyclical_resp_se,
    }).dropna()

    # Within-R² of abs_red ~ gdp_growth
    within_r2s = []
    for iso in country_df["iso3c"]:
        grp = panel[panel["iso3c"] == iso].dropna(subset=["abs_red", "gdp_growth"])
        if len(grp) < 5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                X = add_constant(grp[["gdp_growth"]])
                res = OLS(grp["abs_red"], X).fit()
                within_r2s.append(res.rsquared)
            except Exception:
                pass
    mean_within_r2 = float(np.nanmean(within_r2s)) if within_r2s else 0.0

    # Net Gini level (time-mean)
    net_gini_mean = panel.groupby("iso3c")["gini_net"].mean()
    country_df["net_gini_mean"] = country_df["iso3c"].map(net_gini_mean)

    # VIF computation (cross-sectional)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_struct, vif_cycl = np.nan, np.nan
    try:
        Xvif = country_df[["net_gini_mean", "structural_wedge", "cyclical_resp"]].dropna()
        Xvif_mat = add_constant(Xvif.values)
        vif_struct = variance_inflation_factor(Xvif_mat, 2)
        vif_cycl = variance_inflation_factor(Xvif_mat, 3)
    except Exception as e:
        logger.warning(f"VIF failed: {e}")

    # Pairwise correlations
    corr_mat = country_df[["net_gini_mean", "structural_wedge", "cyclical_resp"]].corr()
    pairwise = {
        "netgini_structural": float(corr_mat.loc["net_gini_mean", "structural_wedge"]),
        "structural_cyclical": float(corr_mat.loc["structural_wedge", "cyclical_resp"]),
        "netgini_cyclical": float(corr_mat.loc["net_gini_mean", "cyclical_resp"]),
    }

    stats = {
        "cyclical_resp_stats": {
            "mean": float(np.nanmean(cyclical_resp)),
            "sd": float(np.nanstd(cyclical_resp)),
            "p25": float(np.nanpercentile(cyclical_resp, 25)),
            "p75": float(np.nanpercentile(cyclical_resp, 75)),
            "pct_negative": float(np.nanmean(cyclical_resp < 0)),
        },
        "structural_wedge_stats": {
            "mean": float(np.nanmean(structural_wedge)),
            "sd": float(np.nanstd(structural_wedge)),
        },
    }

    diagnostics = {
        "n_countries": int(len(country_df)),
        "n_country_years": int(len(panel)),
        "within_sd_wedge_mean": float(mean_within_sd),
        "within_r2_wedge_gdp_mean": float(mean_within_r2),
        "pairwise_corr": pairwise,
        "vif_structural": float(vif_struct) if not np.isnan(vif_struct) else None,
        "vif_cyclical": float(vif_cycl) if not np.isnan(vif_cycl) else None,
        "testability": "OK" if testable else "INSUFFICIENT_WITHIN_VARIATION",
    }

    logger.info(f"Decomposition done. N={len(country_df)} countries.")
    logger.info(f"Cyclical resp: mean={stats['cyclical_resp_stats']['mean']:.3f}, "
                f"sd={stats['cyclical_resp_stats']['sd']:.3f}")
    logger.info(f"Pairwise corr: {pairwise}")

    return {
        "country_df": country_df,
        "diagnostics": diagnostics,
        "stats": stats,
    }
