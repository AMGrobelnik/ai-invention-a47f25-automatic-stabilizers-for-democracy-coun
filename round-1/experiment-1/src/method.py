#!/usr/bin/env python3
"""
Automatic Stabilizers for Democracy:
Structural-Cyclical Wedge Decomposition & Panel Local Projections.

Implements the full pipeline from SWIID/V-Dem/Maddison data download through
Jordà local projections, half-life estimation, triple dissociation, and
leading-indicator tests. Outputs method_out.json in exp_gen_sol_out format.
"""

from __future__ import annotations

import json
import math
import os
import resource
import sys
import warnings
from pathlib import Path

import numpy as np
from loguru import logger

# ── Workspace paths ──────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
OUTPUTS = WORKSPACE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
(WORKSPACE / "logs").mkdir(exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.add(
    sys.stdout, level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
)
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware / memory limits ─────────────────────────────────────────────────
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    import psutil
    return psutil.virtual_memory().total / 1e9


NUM_CPUS = _detect_cpus()
RAM_GB = _container_ram_gb()
RAM_BUDGET = int(min(RAM_GB * 0.75, 30.0) * 1024 ** 3)
logger.info(f"Hardware: {NUM_CPUS} CPUs, {RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
except Exception:
    pass

# ── Add workspace to path ────────────────────────────────────────────────────
sys.path.insert(0, str(WORKSPACE))


# ── Verdict logic ─────────────────────────────────────────────────────────────
def _compute_verdict(lp: dict, halflife: dict, leading: dict, rob: dict) -> tuple[str, str]:
    signals = []

    # Signal 1: delta_h positive and significant at some horizon
    delta_h = [d for d in lp.get("delta_h", []) if not (d is None or np.isnan(d))]
    delta_lo = [d for d in lp.get("delta_h_ci95_lo", []) if not (d is None or np.isnan(d))]
    n_sig_pos = sum(1 for c, lo in zip(delta_h, delta_lo) if c > 0 and lo > 0)
    signals.append(("delta_positive_significant", n_sig_pos >= 2))

    # Signal 2: Wald test favours cyclical over structural at some h
    wald_ps = [p for p in lp.get("dominance_wald_pvalue", []) if p is not None and not np.isnan(p)]
    signals.append(("dominance_wald", any(p < 0.15 for p in wald_ps)))

    # Signal 3: Half-life gradient in cyclical dimension
    hl_low = halflife.get("cyclical_low")
    hl_high = halflife.get("cyclical_high")
    has_gradient = (hl_low is not None and hl_high is not None and
                    not np.isnan(hl_low) and not np.isnan(hl_high) and
                    hl_high < hl_low)
    signals.append(("halflife_gradient", has_gradient))

    # Signal 4: Incremental AUC
    inc_auc = leading.get("incremental_auc")
    signals.append(("incremental_auc", inc_auc is not None and not np.isnan(inc_auc) and inc_auc >= 0.01))

    n_confirm = sum(1 for _, v in signals if v)
    n_total = len(signals)

    reasons = [name for name, v in signals if v]
    missing = [name for name, v in signals if not v]

    if n_confirm >= 3:
        verdict = "CONFIRMS"
        reason = f"Primary hypothesis confirmed on {n_confirm}/{n_total} signals: {reasons}"
    elif n_confirm >= 2:
        verdict = "PARTIAL"
        reason = f"Partial confirmation ({n_confirm}/{n_total} signals). Confirmed: {reasons}. Not confirmed: {missing}"
    elif n_confirm == 0 and delta_h:
        verdict = "DISCONFIRMS"
        reason = f"No confirmatory signals found. Missing: {missing}"
    else:
        verdict = "PARTIAL"
        reason = f"Mixed evidence ({n_confirm}/{n_total}). Confirmed: {reasons}. Missing: {missing}"

    logger.info(f"Verdict: {verdict} | {reason}")
    return verdict, reason


@logger.catch(reraise=True)
def main() -> None:
    logger.info("=" * 70)
    logger.info("STARTING: Automatic Stabilizers for Democracy Pipeline")
    logger.info("=" * 70)

    # ── STEP 0: Data ─────────────────────────────────────────────────────────
    logger.info("STEP 0: Loading and merging data...")
    from steps.s0_data import merge_data
    panel = merge_data()
    n_countries = int(panel["iso3c"].nunique())
    n_cy = len(panel)
    logger.info(f"Panel: {n_countries} countries, {n_cy} country-years")

    if n_countries < 5:
        logger.error("Panel too sparse — fewer than 5 countries. Aborting.")
        sys.exit(1)

    # ── STEP 1: Decomposition ─────────────────────────────────────────────────
    logger.info("STEP 1: Wedge decomposition...")
    from steps.s1_decompose import decompose_wedge
    n_draws = 20 if n_countries > 50 else 100  # scale draws to data size
    decomp = decompose_wedge(panel, n_draws=n_draws)
    country_df = decomp["country_df"]
    diagnostics = decomp["diagnostics"]
    decomp_stats = decomp["stats"]
    logger.info(f"Decomp done: {len(country_df)} countries with estimates")

    # ── STEP 2: Local Projections ─────────────────────────────────────────────
    logger.info("STEP 2: Panel local projections...")
    from steps.s2_local_proj import run_lp

    def _run_lp(p, c):
        return run_lp(p, c, horizons=list(range(7)))

    lp_results = _run_lp(panel, country_df)

    # ── STEP 3: Half-lives ────────────────────────────────────────────────────
    logger.info("STEP 3: Computing half-lives by tercile...")
    from steps.s3_halflife import compute_all_halflives
    halflife_results = compute_all_halflives(lp_results, country_df, panel)

    # ── STEP 4: Triple Dissociation ───────────────────────────────────────────
    logger.info("STEP 4: Triple dissociation tests...")
    from steps.s4_dissociation import run_triple_dissociation
    dissociation = run_triple_dissociation(panel, country_df)

    # ── STEP 5: Leading Indicator ─────────────────────────────────────────────
    logger.info("STEP 5: Leading indicator test...")
    from steps.s5_leading import run_leading_indicator
    leading = run_leading_indicator(panel, country_df)

    # ── STEP 6: Robustness ────────────────────────────────────────────────────
    logger.info("STEP 6: Robustness checks...")
    from steps.s6_robustness import robustness_checks
    robustness = robustness_checks(panel, country_df, _run_lp)

    # ── STEP 7: Figures ───────────────────────────────────────────────────────
    logger.info("STEP 7: Generating figures...")
    from steps.s7_figures import generate_all_figures
    figure_paths = generate_all_figures(lp_results, halflife_results, country_df, panel)
    for k, v in figure_paths.items():
        logger.info(f"  {k}: {v}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict, verdict_reason = _compute_verdict(lp_results, halflife_results, leading, robustness)

    # ── Build method_out.json (exp_gen_sol_out schema) ────────────────────────
    logger.info("Building method_out.json...")

    # Helper: safe float
    def sf(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if np.isnan(f) or np.isinf(f) else f
        except (TypeError, ValueError):
            return None

    def sl(lst):
        return [sf(v) for v in (lst or [])]

    # Build examples per country (wedge decomposition results)
    examples_decomp = []
    for _, row in country_df.iterrows():
        iso = row["iso3c"]
        cy_resp = sf(row.get("cyclical_resp"))
        sw = sf(row.get("structural_wedge"))
        ng = sf(row.get("net_gini_mean"))

        # Country panel summary
        cp = panel[panel["iso3c"] == iso]
        ymin = int(cp["year"].min()) if len(cp) else 1990
        ymax = int(cp["year"].max()) if len(cp) else 2022
        n_obs = len(cp)
        mean_dem = sf(cp["v2x_libdem"].mean()) if "v2x_libdem" in cp.columns else None
        erosion = bool(cp["secular_erosion"].any()) if "secular_erosion" in cp.columns else False

        input_str = (
            f"Country: {iso} | Years: {ymin}–{ymax} | N={n_obs} obs | "
            f"Mean v2x_libdem: {mean_dem:.3f}" if mean_dem else f"Country: {iso}"
        )
        output_str = json.dumps({
            "structural_wedge": sw,
            "cyclical_resp": cy_resp,
            "cyclical_resp_se": sf(row.get("cyclical_resp_se")),
            "net_gini_mean": ng,
            "secular_erosion": erosion,
        })
        example = {
            "input": input_str,
            "output": output_str,
            "predict_cyclical_resp": str(cy_resp) if cy_resp is not None else "",
            "predict_structural_wedge": str(sw) if sw is not None else "",
            "metadata_iso3c": iso,
            "metadata_n_obs": n_obs,
        }
        examples_decomp.append(example)

    # LP results as examples
    examples_lp = []
    for i, h in enumerate(lp_results["horizons"]):
        delta = sf(lp_results["delta_h"][i]) if i < len(lp_results["delta_h"]) else None
        phi = sf(lp_results["phi_h"][i]) if i < len(lp_results["phi_h"]) else None
        theta = sf(lp_results["theta_h"][i]) if i < len(lp_results["theta_h"]) else None
        beta = sf(lp_results["beta_h"][i]) if i < len(lp_results.get("beta_h", [])) else None
        wald_p = sf(lp_results["dominance_wald_pvalue"][i]) if i < len(lp_results["dominance_wald_pvalue"]) else None

        input_str = f"Local Projection | horizon h={h} | outcome=v2x_libdem | shock=gdp_innovation"
        output_str = json.dumps({
            "h": h,
            "beta_h": beta,
            "delta_h": delta,
            "delta_h_ci95_lo": sf(lp_results["delta_h_ci95_lo"][i]) if i < len(lp_results["delta_h_ci95_lo"]) else None,
            "delta_h_ci95_hi": sf(lp_results["delta_h_ci95_hi"][i]) if i < len(lp_results["delta_h_ci95_hi"]) else None,
            "phi_h": phi,
            "theta_h": theta,
            "dominance_wald_pvalue": wald_p,
            "n_obs": lp_results.get("n_obs_per_h", [None] * (i + 1))[i] if i < len(lp_results.get("n_obs_per_h", [])) else None,
        })
        examples_lp.append({
            "input": input_str,
            "output": output_str,
            "predict_delta_h": str(delta) if delta is not None else "",
            "predict_phi_h": str(phi) if phi is not None else "",
            "metadata_horizon": h,
        })

    # Aggregate summary example
    summary_example = {
        "input": "Aggregate summary: stabilizer decomposition, LP, half-life, leading indicator",
        "output": json.dumps({
            "step0_diagnostics": {
                "n_countries": diagnostics["n_countries"],
                "n_country_years": diagnostics["n_country_years"],
                "within_sd_wedge_mean": sf(diagnostics["within_sd_wedge_mean"]),
                "within_r2_wedge_gdp_mean": sf(diagnostics["within_r2_wedge_gdp_mean"]),
                "pairwise_corr": {k: sf(v) for k, v in diagnostics["pairwise_corr"].items()},
                "vif_structural": sf(diagnostics.get("vif_structural")),
                "vif_cyclical": sf(diagnostics.get("vif_cyclical")),
                "testability": diagnostics["testability"],
            },
            "step1_decomposition": {
                "cyclical_resp_stats": {k: sf(v) for k, v in decomp_stats["cyclical_resp_stats"].items()},
                "structural_wedge_stats": {k: sf(v) for k, v in decomp_stats["structural_wedge_stats"].items()},
            },
            "step2_lp": {
                "horizons": lp_results["horizons"],
                "delta_h": sl(lp_results["delta_h"]),
                "delta_h_ci95_lo": sl(lp_results["delta_h_ci95_lo"]),
                "delta_h_ci95_hi": sl(lp_results["delta_h_ci95_hi"]),
                "phi_h": sl(lp_results["phi_h"]),
                "theta_h": sl(lp_results["theta_h"]),
                "dominance_wald_pvalue": sl(lp_results["dominance_wald_pvalue"]),
                "n_obs": lp_results.get("n_obs", 0),
            },
            "step3_halflife": {k: sf(v) for k, v in halflife_results.items()},
            "step4_triple_dissociation": {
                "polynomial_delta_h_significant": dissociation.get("polynomial_delta_h_significant"),
                "cem_n_matched": dissociation.get("cem_n_matched", 0),
                "cem_delta_h_significant": dissociation.get("cem_delta_h_significant"),
            },
            "step5_leading_indicator": {
                "baseline_auc": sf(leading.get("baseline_auc")),
                "augmented_auc": sf(leading.get("augmented_auc")),
                "incremental_auc": sf(leading.get("incremental_auc")),
                "cyclical_resp_logit_coef": sf(leading.get("cyclical_resp_logit_coef")),
                "cyclical_resp_logit_pvalue": sf(leading.get("cyclical_resp_logit_pvalue")),
            },
            "step6_robustness": {
                "without_secular_erosion_delta_h_peak": sf(robustness.get("without_secular_erosion_delta_h_peak")),
                "secular_erosion_only_delta_h_peak": sf(robustness.get("secular_erosion_only_delta_h_peak")),
                "placebo_shuffle_delta_h_peak": sf(robustness.get("placebo_shuffle_delta_h_peak")),
                "mi_combined_se_vs_single_draw_ratio": sf(robustness.get("mi_combined_se_vs_single_draw_ratio")),
            },
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "figures": figure_paths,
        }),
        "metadata_n_countries": n_countries,
        "metadata_verdict": verdict,
    }

    # Assemble final output
    method_out = {
        "metadata": {
            "method_name": "AutoStabilizersForDemocracy",
            "description": (
                "Structural-cyclical wedge decomposition of redistribution + panel local projections "
                "(Jordà 2005) to test whether counter-cyclical redistribution buffers democratic backsliding"
            ),
            "n_mi_draws": n_draws,
            "outcome": "v2x_libdem",
            "shock": "gdp_innovation",
            "horizons": lp_results["horizons"],
            "verdict": verdict,
        },
        "datasets": [
            {
                "dataset": "wedge_decomposition",
                "examples": examples_decomp,
            },
            {
                "dataset": "local_projections",
                "examples": examples_lp,
            },
            {
                "dataset": "aggregate_summary",
                "examples": [summary_example],
            },
        ],
    }

    out_path = OUTPUTS / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2))
    logger.info(f"Saved method_out.json ({out_path.stat().st_size / 1e3:.0f} KB)")

    logger.info("=" * 70)
    logger.info(f"DONE — Verdict: {verdict}")
    logger.info(f"Reason: {verdict_reason}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
