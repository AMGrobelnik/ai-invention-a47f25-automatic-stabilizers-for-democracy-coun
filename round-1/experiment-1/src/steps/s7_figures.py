"""Step 7: Generate all figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIG_DIR = Path(__file__).parent.parent / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig1_lp_interactions(lp_results: Dict) -> Path:
    """3-panel: delta_h, phi_h, theta_h across horizons."""
    horizons = lp_results["horizons"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    specs = [
        ("delta_h", "Cyclical Resp. (δ_h)", "steelblue"),
        ("phi_h", "Structural Wedge (φ_h)", "coral"),
        ("theta_h", "Net Gini Level (θ_h)", "seagreen"),
    ]
    for ax, (key, label, color) in zip(axes, specs):
        coefs = np.array(lp_results.get(key, [np.nan] * len(horizons)), dtype=float)
        lo = np.array(lp_results.get(f"{key}_ci95_lo", coefs - 0.1), dtype=float)
        hi = np.array(lp_results.get(f"{key}_ci95_hi", coefs + 0.1), dtype=float)
        ax.plot(horizons, coefs, "o-", color=color, lw=2, label=label)
        ax.fill_between(horizons, lo, hi, alpha=0.2, color=color)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Horizon h (years)")
        ax.set_ylabel("Coefficient")
        ax.set_title(label)
        ax.set_xticks(horizons)
        ax.grid(alpha=0.3)

    fig.suptitle("Panel LP Interaction Coefficients (shock × moderator)", fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / "fig1_lp_interactions.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig2_halflife_table(halflife_results: Dict) -> Path:
    """Bar chart of half-lives by tercile × moderator dimension."""
    dims = [
        ("cyclical", "Cyclical\nResponsiveness", "steelblue"),
        ("structural", "Structural\nWedge", "coral"),
        ("netgini", "Net Gini\nLevel", "seagreen"),
    ]
    terciles = ["low", "median", "high"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(terciles))
    width = 0.25

    for i, (dim, label, color) in enumerate(dims):
        vals = [halflife_results.get(f"{dim}_{t}") for t in terciles]
        vals_plot = [v if v is not None and not np.isnan(v) else 0 for v in vals]
        bars = ax.bar(x + i * width, vals_plot, width, label=label, color=color, alpha=0.8)
        for bar, v in zip(bars, vals):
            if v is not None and not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Low Tercile", "Median", "High Tercile"])
    ax.set_ylabel("Recovery Half-life (years)")
    ax.set_title("Recovery Half-life by Moderator Dimension and Tercile")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIG_DIR / "fig2_halflife_table.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig3_irfs_by_tercile(lp_results: Dict, country_df) -> Path:
    """IRF curves overlaid for low/med/high cyclical_resp tercile."""
    horizons = np.array(lp_results["horizons"], dtype=float)
    beta_h = np.array(lp_results.get("beta_h", [0] * len(horizons)), dtype=float)
    delta_h = np.array(lp_results.get("delta_h", [0] * len(horizons)), dtype=float)
    phi_h = np.array(lp_results.get("phi_h", [0] * len(horizons)), dtype=float)

    cyc_vals = country_df["cyclical_resp"].dropna().values
    q33, q67 = np.nanpercentile(cyc_vals, [33, 67])
    low = np.nanmedian(cyc_vals[cyc_vals <= q33])
    med = np.nanmedian(cyc_vals)
    high = np.nanmedian(cyc_vals[cyc_vals >= q67])
    struct_med = country_df["structural_wedge"].median()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Low": "coral", "Median": "gray", "High": "steelblue"}
    for label, cyc_val in [("Low", low), ("Median", med), ("High", high)]:
        irf = (beta_h + np.nan_to_num(delta_h) * cyc_val + np.nan_to_num(phi_h) * struct_med) * (-1.0)
        ax.plot(horizons, irf, "o-", label=f"{label} cyclical resp.", color=colors[label], lw=2)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Horizon h (years)")
    ax.set_ylabel("y_{t+h} - y_{t-1} (democracy change)")
    ax.set_title("IRFs to -1σ GDP Shock by Cyclical Responsiveness Tercile")
    ax.legend()
    ax.set_xticks(horizons)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = FIG_DIR / "fig3_irfs_by_tercile.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig4_auroc(panel: "pd.DataFrame", country_df) -> Path:
    """ROC curves for baseline vs augmented backsliding predictor."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import StandardScaler
    import warnings

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    try:
        from steps.s5_leading import define_backsliding
        panel_bs = define_backsliding(panel)
        if "secular_erosion" in panel_bs.columns:
            panel_bs = panel_bs[~panel_bs["secular_erosion"]]
        panel_bs = panel_bs.merge(country_df[["iso3c", "cyclical_resp"]], on="iso3c", how="left")
        panel_bs["v2x_libdem_lag"] = panel_bs.groupby("iso3c")["v2x_libdem"].shift(1)
        panel_bs["log_gdppc_lag"] = panel_bs.groupby("iso3c")["log_gdppc"].shift(1)
        panel_bs["gini_net_lag"] = panel_bs.groupby("iso3c")["gini_net"].shift(1)
        df_m = panel_bs[["backsliding", "v2x_libdem_lag", "log_gdppc_lag",
                          "gini_net_lag", "cyclical_resp"]].dropna()

        if len(df_m) > 50 and df_m["backsliding"].sum() >= 5:
            y = df_m["backsliding"].values
            X_base = df_m[["v2x_libdem_lag", "log_gdppc_lag", "gini_net_lag"]].values
            X_aug = df_m[["v2x_libdem_lag", "log_gdppc_lag", "gini_net_lag", "cyclical_resp"]].values

            for X_data, label, color in [
                (X_base, "Baseline", "coral"),
                (X_aug, "Augmented (+Cyclical Resp.)", "steelblue"),
            ]:
                scaler = StandardScaler()
                X_s = scaler.fit_transform(X_data)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
                    clf.fit(X_s, y)
                proba = clf.predict_proba(X_s)[:, 1]
                fpr, tpr, _ = roc_curve(y, proba)
                auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC={auc_val:.3f})")
    except Exception as e:
        ax.text(0.5, 0.5, f"ROC unavailable:\n{e}", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Backsliding Prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = FIG_DIR / "fig4_auroc.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig5_country_scatter(country_df, halflife_per_country: dict = None) -> Path:
    """Scatter: cyclical_resp vs mean half-life across countries."""
    fig, ax = plt.subplots(figsize=(8, 6))

    cyc = country_df["cyclical_resp"].values
    struct = country_df["structural_wedge"].values
    isos = country_df["iso3c"].values

    scatter = ax.scatter(cyc, struct, c=cyc, cmap="RdYlGn", s=60, alpha=0.7, edgecolors="k", lw=0.3)
    plt.colorbar(scatter, ax=ax, label="Cyclical Responsiveness")

    # Label notable countries
    notable = {"USA", "DEU", "SWE", "BRA", "POL", "KOR", "HUN", "TUR", "VEN"}
    for iso, x, y in zip(isos, cyc, struct):
        if iso in notable:
            ax.annotate(iso, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Cyclical Responsiveness (−β_GDP)")
    ax.set_ylabel("Structural Wedge (time-mean abs. redistribution)")
    ax.set_title("Country Decomposition: Cyclical vs. Structural Redistribution")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = FIG_DIR / "fig5_country_scatter.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_figures(lp_results: Dict, halflife_results: Dict, country_df, panel) -> Dict:
    paths = {}
    try:
        paths["fig1"] = str(fig1_lp_interactions(lp_results))
    except Exception as e:
        paths["fig1"] = f"FAILED: {e}"
    try:
        paths["fig2"] = str(fig2_halflife_table(halflife_results))
    except Exception as e:
        paths["fig2"] = f"FAILED: {e}"
    try:
        paths["fig3"] = str(fig3_irfs_by_tercile(lp_results, country_df))
    except Exception as e:
        paths["fig3"] = f"FAILED: {e}"
    try:
        paths["fig4"] = str(fig4_auroc(panel, country_df))
    except Exception as e:
        paths["fig4"] = f"FAILED: {e}"
    try:
        paths["fig5"] = str(fig5_country_scatter(country_df))
    except Exception as e:
        paths["fig5"] = f"FAILED: {e}"
    return paths
