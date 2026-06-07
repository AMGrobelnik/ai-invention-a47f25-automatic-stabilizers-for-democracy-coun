"""Step 3: Impulse-response half-life computation by moderator tercile."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.optimize import curve_fit


def _exponential_decay(h: np.ndarray, A: float, lam: float) -> np.ndarray:
    return A * np.exp(-lam * h)


def _fit_halflife(irf: np.ndarray, horizons: np.ndarray) -> float:
    """Fit exponential decay to IRF and return half-life = log(2)/lambda."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _exponential_decay, horizons, irf,
                p0=[-0.05, 0.3], maxfev=2000,
                bounds=([-np.inf, 0.001], [np.inf, 10.0])
            )
        lam = popt[1]
        if lam <= 0:
            return np.nan
        return float(np.log(2) / lam)
    except Exception:
        # Linear interpolation fallback
        try:
            if irf[0] == 0:
                return np.nan
            half_val = irf[0] / 2
            for i in range(1, len(irf)):
                if irf[i] >= half_val:
                    return float(horizons[i])
            return float(horizons[-1] * 2)
        except Exception:
            return np.nan


def compute_irfs(
    lp_results: Dict,
    country_df,
    panel,
    moderator: str = "cyclical_resp",
) -> Dict[str, float]:
    """
    Compute IRF for low/median/high tercile of moderator,
    return half-lives.
    """
    horizons = np.array(lp_results["horizons"], dtype=float)
    beta_h = np.array(lp_results["beta_h"], dtype=float)
    delta_h = np.array(lp_results["delta_h"], dtype=float)
    phi_h = np.array(lp_results["phi_h"], dtype=float)
    theta_h = np.array(lp_results["theta_h"], dtype=float)

    # Get tercile breakpoints for the moderator
    mod_vals = country_df[moderator].dropna().values
    q33, q67 = np.nanpercentile(mod_vals, [33, 67])

    low_val = np.nanpercentile(mod_vals[mod_vals <= q33], 50)
    med_val = np.nanpercentile(mod_vals, 50)
    high_val = np.nanpercentile(mod_vals[mod_vals >= q67], 50)

    # Other moderators at median
    struct_med = country_df["structural_wedge"].median()
    netgini_med = panel["gini_net"].median() if panel is not None else 40.0
    cyc_med = country_df["cyclical_resp"].median()

    def irf_at(cyc_val: float, struct_val: float, ng_val: float) -> np.ndarray:
        irf = np.zeros(len(horizons))
        for i, h in enumerate(horizons):
            if np.isnan(beta_h[i]):
                irf[i] = 0.0
                continue
            val = (
                beta_h[i] * (-1.0)  # shock = -1 SD
                + (delta_h[i] if not np.isnan(delta_h[i]) else 0) * (-1.0) * cyc_val
                + (phi_h[i] if not np.isnan(phi_h[i]) else 0) * (-1.0) * struct_val
                + (theta_h[i] if not np.isnan(theta_h[i]) else 0) * (-1.0) * ng_val
            )
            irf[i] = val
        return irf

    results: Dict[str, float] = {}

    for tercile_name, tercile_val in [("low", low_val), ("median", med_val), ("high", high_val)]:
        if moderator == "cyclical_resp":
            irf = irf_at(tercile_val, struct_med, netgini_med)
        elif moderator == "structural_wedge":
            irf = irf_at(cyc_med, tercile_val, netgini_med)
        else:  # net_gini
            irf = irf_at(cyc_med, struct_med, tercile_val)

        hl = _fit_halflife(irf, horizons)
        results[tercile_name] = hl
        logger.info(f"  {moderator} {tercile_name} (val={tercile_val:.3f}): half-life={hl:.2f}")

    return results


def compute_all_halflives(lp_results: Dict, country_df, panel) -> Dict:
    """Compute half-lives for all three moderator dimensions."""
    logger.info("Computing half-lives by moderator tercile...")

    out: Dict = {}
    for mod in ["cyclical_resp", "structural_wedge"]:
        try:
            hl = compute_irfs(lp_results, country_df, panel, moderator=mod)
            for k, v in hl.items():
                out[f"{mod.replace('_resp', '').replace('_wedge', '')}_{k}"] = v
        except Exception as e:
            logger.warning(f"Half-life computation failed for {mod}: {e}")
            for k in ["low", "median", "high"]:
                out[f"{mod.replace('_resp', '').replace('_wedge', '')}_{k}"] = np.nan

    # Net Gini dimension
    try:
        # Add net gini to country_df temporarily
        ng_country = panel.groupby("iso3c")["gini_net"].median().reset_index()
        ng_country.columns = ["iso3c", "net_gini_country"]
        cdf_tmp = country_df.merge(ng_country, on="iso3c", how="left")
        cdf_tmp = cdf_tmp.rename(columns={"net_gini_country": "net_gini"})
        hl = compute_irfs(lp_results, cdf_tmp.rename(columns={"net_gini": "cyclical_resp"}
                          ).assign(cyclical_resp=cdf_tmp.get("net_gini_country", cdf_tmp.get("net_gini_mean", 40.0))),
                          panel, moderator="cyclical_resp")
        # Not the cleanest but avoids duplicate code
        ng_hl = compute_irfs(lp_results, cdf_tmp.rename(columns={"net_gini": "structural_wedge"}
                             ).assign(structural_wedge=cdf_tmp.get("net_gini_country",
                                                                    cdf_tmp.get("net_gini_mean", 40.0))),
                             panel, moderator="structural_wedge")
        out["netgini_low"] = ng_hl.get("low", np.nan)
        out["netgini_median"] = ng_hl.get("median", np.nan)
        out["netgini_high"] = ng_hl.get("high", np.nan)
    except Exception as e:
        logger.warning(f"Net Gini half-life failed: {e}")
        for k in ["low", "median", "high"]:
            out[f"netgini_{k}"] = np.nan

    # Rename for output schema
    renamed = {}
    for k, v in out.items():
        k2 = k.replace("cyclical_", "cyclical_").replace("structural_", "structural_")
        renamed[k2] = float(v) if not np.isnan(v) else None
    return renamed
