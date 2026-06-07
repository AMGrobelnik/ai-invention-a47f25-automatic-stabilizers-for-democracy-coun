"""Step 5: Leading indicator test — backsliding prediction via logit + AUROC."""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def define_backsliding(panel: pd.DataFrame, threshold: float = 0.05,
                        horizon: int = 5) -> pd.DataFrame:
    """
    Label each country-year: 1 if v2x_libdem drops >= threshold within next horizon years.
    Exclude secular erosion countries.
    """
    panel = panel.copy()
    panel["backsliding"] = 0

    for iso, grp in panel.groupby("iso3c"):
        grp = grp.sort_values("year").set_index("year")
        for yr in grp.index:
            future_years = [y for y in range(yr + 1, yr + horizon + 1) if y in grp.index]
            if not future_years or "v2x_libdem" not in grp.columns:
                continue
            base = grp.loc[yr, "v2x_libdem"]
            future_min = grp.loc[future_years, "v2x_libdem"].min()
            if pd.isna(base) or pd.isna(future_min):
                continue
            if future_min <= base - threshold:
                panel.loc[(panel["iso3c"] == iso) & (panel["year"] == yr), "backsliding"] = 1

    n_episodes = panel["backsliding"].sum()
    logger.info(f"Backsliding episodes defined: {n_episodes} / {len(panel)}")
    return panel


def run_leading_indicator(panel: pd.DataFrame, country_df: pd.DataFrame) -> Dict:
    """Logit + AUROC for backsliding prediction."""
    logger.info("Running leading indicator test...")

    panel = define_backsliding(panel)

    # Exclude secular erosion countries
    panel_clean = panel[~panel["secular_erosion"]].copy() if "secular_erosion" in panel.columns else panel.copy()

    # Merge country-level features
    panel_clean = panel_clean.merge(
        country_df[["iso3c", "cyclical_resp", "structural_wedge"]],
        on="iso3c", how="left"
    )

    # Feature set
    baseline_features = ["v2x_libdem", "log_gdppc", "gini_net"]
    optional_features = ["v2juncind", "v2x_rule", "v2cacamps"]
    augmented_features = ["cyclical_resp"]

    # Lag democracy features by 1
    panel_clean["v2x_libdem_lag"] = panel_clean.groupby("iso3c")["v2x_libdem"].shift(1)
    panel_clean["log_gdppc_lag"] = panel_clean.groupby("iso3c")["log_gdppc"].shift(1)
    panel_clean["gini_net_lag"] = panel_clean.groupby("iso3c")["gini_net"].shift(1)

    base_cols = ["v2x_libdem_lag", "log_gdppc_lag", "gini_net_lag"]
    for c in optional_features:
        if c in panel_clean.columns:
            lag_c = f"{c}_lag"
            panel_clean[lag_c] = panel_clean.groupby("iso3c")[c].shift(1)
            base_cols.append(lag_c)

    # Region FE (one-hot)
    region_map = {
        "USA": "NAM", "CAN": "NAM", "MEX": "LAM", "BRA": "LAM", "ARG": "LAM",
        "COL": "LAM", "CHL": "LAM", "PER": "LAM", "ECU": "LAM", "BOL": "LAM",
        "URY": "LAM", "PRY": "LAM", "CRI": "LAM", "PAN": "LAM", "DOM": "LAM",
        "JAM": "LAM", "DEU": "EUR", "FRA": "EUR", "GBR": "EUR", "ITA": "EUR",
        "ESP": "EUR", "PRT": "EUR", "NLD": "EUR", "BEL": "EUR", "AUT": "EUR",
        "CHE": "EUR", "SWE": "EUR", "NOR": "EUR", "DNK": "EUR", "FIN": "EUR",
        "IRL": "EUR", "POL": "EUR", "CZE": "EUR", "HUN": "EUR", "SVK": "EUR",
        "ROU": "EUR", "BGR": "EUR", "HRV": "EUR", "SVN": "EUR", "LTU": "EUR",
        "LVA": "EUR", "EST": "EUR", "GRC": "EUR", "TUR": "EUR",
        "RUS": "EUR", "UKR": "EUR",
        "CHN": "ASI", "JPN": "ASI", "KOR": "ASI", "IND": "ASI", "IDN": "ASI",
        "PHL": "ASI", "THA": "ASI", "MYS": "ASI", "VNM": "ASI", "PAK": "ASI",
        "BGD": "ASI", "LKA": "ASI", "TWN": "ASI", "HKG": "ASI", "SGP": "ASI",
        "AUS": "OCE", "NZL": "OCE",
        "NGA": "AFR", "KEN": "AFR", "GHA": "AFR", "ETH": "AFR", "TZA": "AFR",
        "UGA": "AFR", "ZAF": "AFR", "MAR": "AFR", "DZA": "AFR", "EGY": "AFR",
        "TUN": "AFR",
        "ISR": "MNA", "JOR": "MNA", "LBN": "MNA", "VEN": "LAM",
    }
    panel_clean["region"] = panel_clean["iso3c"].map(region_map).fillna("OTH")
    region_dummies = pd.get_dummies(panel_clean["region"], prefix="reg", drop_first=True)

    all_cols = base_cols + ["cyclical_resp"]
    df_model = panel_clean[["backsliding"] + all_cols].join(region_dummies)
    df_model = df_model.dropna()

    if len(df_model) < 50 or df_model["backsliding"].sum() < 5:
        logger.warning(f"Too few observations ({len(df_model)}) or episodes ({df_model['backsliding'].sum()})")
        return {
            "baseline_auc": None, "augmented_auc": None,
            "incremental_auc": None,
            "cyclical_resp_logit_coef": None,
            "cyclical_resp_logit_pvalue": None,
        }

    y = df_model["backsliding"].values
    X_base = df_model[base_cols + list(region_dummies.columns)].values
    X_aug = df_model[all_cols + list(region_dummies.columns)].values

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=min(5, df_model["backsliding"].sum()), shuffle=True, random_state=42)

    def cv_auc(X: np.ndarray) -> float:
        aucs = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
                clf.fit(X_tr, y_tr)
            if y_te.sum() == 0 or y_te.sum() == len(y_te):
                continue
            proba = clf.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, proba))
        return float(np.mean(aucs)) if aucs else np.nan

    baseline_auc = cv_auc(X_base)
    augmented_auc = cv_auc(X_aug)
    incremental_auc = augmented_auc - baseline_auc if not np.isnan(augmented_auc) else np.nan

    # Full-sample logit for coefficient
    scaler = StandardScaler()
    X_aug_scaled = scaler.fit_transform(X_aug)
    clf_full = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf_full.fit(X_aug_scaled, y)
    feat_names = all_cols + list(region_dummies.columns)
    cyc_idx = feat_names.index("cyclical_resp")
    cyc_coef = float(clf_full.coef_[0][cyc_idx])

    # Approximate p-value via Wald test
    from scipy.stats import norm
    n = len(y)
    p_hat = clf_full.predict_proba(X_aug_scaled)[:, 1]
    W = np.diag(p_hat * (1 - p_hat))
    XtWX = X_aug_scaled.T @ W @ X_aug_scaled
    try:
        cov = np.linalg.pinv(XtWX)
        cyc_se = np.sqrt(max(cov[cyc_idx, cyc_idx], 1e-10))
        z = cyc_coef / cyc_se
        cyc_pval = float(2 * norm.sf(abs(z)))
    except Exception:
        cyc_pval = np.nan

    logger.info(f"Baseline AUC: {baseline_auc:.4f}, Augmented AUC: {augmented_auc:.4f}, "
                f"Increment: {incremental_auc:.4f}")
    logger.info(f"cyclical_resp coef: {cyc_coef:.4f}, p={cyc_pval:.4f}")

    return {
        "baseline_auc": float(baseline_auc),
        "augmented_auc": float(augmented_auc),
        "incremental_auc": float(incremental_auc),
        "cyclical_resp_logit_coef": float(cyc_coef),
        "cyclical_resp_logit_pvalue": float(cyc_pval),
    }
