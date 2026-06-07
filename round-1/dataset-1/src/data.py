#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "pyarrow>=15", "loguru>=0.7"]
# ///
"""Convert Democratic Resilience Panel datasets to exp_sel_data_out schema.

Each data ROW = one example with input (predictor features JSON string) and output (target as string).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WS = Path(__file__).parent
DS = WS / "temp" / "datasets"
OWID_TB = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables")


def row_to_example(row: dict, input_cols: list[str], output_col: str, meta: dict | None = None) -> dict:
    inp = {k: (None if pd.isna(v) else v) for k, v in row.items() if k in input_cols}
    out_val = row.get(output_col)
    example: dict = {
        "input": json.dumps(inp, default=str),
        "output": str(round(float(out_val), 6)) if out_val is not None and not pd.isna(out_val) else "",
    }
    if meta:
        example.update(meta)
    return example


def df_to_examples(df: pd.DataFrame, input_cols: list[str], output_col: str, meta_cols: list[str] | None = None) -> list[dict]:
    examples = []
    for idx, row in df.iterrows():
        row_d = row.to_dict()
        meta: dict = {"metadata_row_index": int(idx)}
        if meta_cols:
            for mc in meta_cols:
                if mc in row_d and pd.notna(row_d[mc]):
                    meta[f"metadata_{mc}"] = row_d[mc]
        example = row_to_example(row_d, input_cols=input_cols, output_col=output_col, meta=meta)
        examples.append(example)
    return examples


def load_panel_primary() -> list[dict]:
    df = pd.read_parquet(WS / "panel_primary.parquet")
    df = df[df["year"] >= 1990].copy()
    input_cols = [c for c in df.columns if c not in ["gini_disp", "country"]]
    examples = df_to_examples(
        df=df,
        input_cols=input_cols,
        output_col="gini_disp",
        meta_cols=["iso3", "year"],
    )
    logger.info(f"panel_primary: {len(examples)} examples")
    return examples


def load_panel_full() -> list[dict]:
    df = pd.read_parquet(WS / "panel_full.parquet")
    df = df[df["year"] >= 1990].copy()
    input_cols = [c for c in df.columns if c not in ["gini_disp", "country"]]
    examples = df_to_examples(
        df=df,
        input_cols=input_cols,
        output_col="gini_disp",
        meta_cols=["iso3", "year"],
    )
    logger.info(f"panel_full: {len(examples)} examples")
    return examples


def load_swiid_summary() -> list[dict]:
    df = pd.read_csv(DS / "swiid_summary.csv")
    df = df[df["year"] >= 1990].dropna(subset=["gini_disp"]).copy()
    input_cols = ["country", "year", "gini_mkt", "gini_mkt_se", "gini_disp_se", "abs_red", "rel_red"]
    examples = df_to_examples(
        df=df,
        input_cols=[c for c in input_cols if c in df.columns],
        output_col="gini_disp",
        meta_cols=["country", "year"],
    )
    logger.info(f"swiid_summary: {len(examples)} examples")
    return examples


def load_vdem_polyarchy() -> list[dict]:
    df = pd.read_csv(DS / "owid_vdem_polyarchy.csv")
    df = df.rename(columns={"Entity": "country", "Code": "iso3", "Year": "year", "Electoral democracy index": "v2x_polyarchy"})
    df = df[df["year"] >= 1990].dropna(subset=["iso3", "v2x_polyarchy"]).copy()
    df = df[df["iso3"].str.len() == 3]
    input_cols = ["country", "iso3", "year"]
    examples = df_to_examples(
        df=df,
        input_cols=input_cols,
        output_col="v2x_polyarchy",
        meta_cols=["iso3", "year"],
    )
    logger.info(f"vdem_polyarchy: {len(examples)} examples")
    return examples


def load_vdem_libdem() -> list[dict]:
    df = pd.read_csv(DS / "owid_vdem_libdem.csv")
    df = df.rename(columns={"Entity": "country", "Code": "iso3", "Year": "year", "Liberal democracy index": "v2x_libdem"})
    df = df[df["year"] >= 1990].dropna(subset=["iso3", "v2x_libdem"]).copy()
    df = df[df["iso3"].str.len() == 3]
    input_cols = ["country", "iso3", "year"]
    examples = df_to_examples(
        df=df,
        input_cols=input_cols,
        output_col="v2x_libdem",
        meta_cols=["iso3", "year"],
    )
    logger.info(f"vdem_libdem: {len(examples)} examples")
    return examples


def load_maddison() -> list[dict]:
    path = OWID_TB / "full_garden_ggdc_2024-04-26_maddison_project_database_maddison_project_database.json"
    raw = json.loads(path.read_text())
    df = pd.DataFrame(raw)
    df = df[df["year"] >= 1990].dropna(subset=["gdp_per_capita"]).copy()
    input_cols = ["country", "year", "region", "population"]
    examples = df_to_examples(
        df=df,
        input_cols=[c for c in input_cols if c in df.columns],
        output_col="gdp_per_capita",
        meta_cols=["country", "year"],
    )
    logger.info(f"maddison_gdppc: {len(examples)} examples")
    return examples


def load_wb_gdp_growth() -> list[dict]:
    raw = json.loads((DS / "wb_gdp_growth_raw.json").read_text())
    rows = raw[1] if isinstance(raw, list) and len(raw) > 1 and raw[1] else []
    records = []
    for r in rows:
        iso3 = r.get("countryiso3code", "")
        year = r.get("date")
        val = r.get("value")
        if iso3 and len(iso3) == 3 and year and val is not None:
            try:
                records.append({"iso3": iso3, "country": r.get("country", {}).get("value", ""), "year": int(year), "gdp_growth": float(val)})
            except (ValueError, TypeError):
                pass
    df = pd.DataFrame(records)
    df = df[df["year"] >= 1990].copy()
    input_cols = ["country", "iso3", "year"]
    examples = df_to_examples(
        df=df,
        input_cols=input_cols,
        output_col="gdp_growth",
        meta_cols=["iso3", "year"],
    )
    logger.info(f"wb_gdp_growth: {len(examples)} examples")
    return examples


def load_wb_tot() -> list[dict]:
    raw = json.loads((DS / "wb_tot_raw.json").read_text())
    rows = raw[1] if isinstance(raw, list) and len(raw) > 1 and raw[1] else []
    records = []
    for r in rows:
        iso3 = r.get("countryiso3code", "")
        year = r.get("date")
        val = r.get("value")
        if iso3 and len(iso3) == 3 and year and val is not None:
            try:
                records.append({"iso3": iso3, "country": r.get("country", {}).get("value", ""), "year": int(year), "terms_of_trade": float(val)})
            except (ValueError, TypeError):
                pass
    df = pd.DataFrame(records)
    df = df[df["year"] >= 1990].copy()
    input_cols = ["country", "iso3", "year"]
    examples = df_to_examples(
        df=df,
        input_cols=input_cols,
        output_col="terms_of_trade",
        meta_cols=["iso3", "year"],
    )
    logger.info(f"wb_tot: {len(examples)} examples")
    return examples


def load_freedom_house() -> list[dict]:
    path = OWID_TB / "full_garden_democracy_2025-06-02_fh_fh.json"
    raw = json.loads(path.read_text())
    df = pd.DataFrame(raw)
    df = df[df["year"] >= 1990].dropna(subset=["polrights", "civlibs"]).copy()
    df["fh_combined"] = df["polrights"].astype(float) + df["civlibs"].astype(float)
    input_cols = ["country", "year", "polrights", "civlibs", "regime"]
    examples = df_to_examples(
        df=df,
        input_cols=[c for c in input_cols if c in df.columns],
        output_col="fh_combined",
        meta_cols=["country", "year"],
    )
    logger.info(f"freedom_house: {len(examples)} examples")
    return examples


def load_wb_gdppc_ppp() -> list[dict]:
    raw = json.loads((DS / "wb_gdppc_ppp_raw.json").read_text())
    rows = raw[1] if isinstance(raw, list) and len(raw) > 1 and raw[1] else []
    records = []
    for r in rows:
        iso3 = r.get("countryiso3code", "")
        year = r.get("date")
        val = r.get("value")
        if iso3 and len(iso3) == 3 and year and val is not None:
            try:
                records.append({"iso3": iso3, "country": r.get("country", {}).get("value", ""), "year": int(year), "gdppc_ppp": float(val)})
            except (ValueError, TypeError):
                pass
    df = pd.DataFrame(records)
    df = df[df["year"] >= 1990].copy()
    input_cols = ["country", "iso3", "year"]
    examples = df_to_examples(
        df=df,
        input_cols=input_cols,
        output_col="gdppc_ppp",
        meta_cols=["iso3", "year"],
    )
    logger.info(f"wb_gdppc_ppp: {len(examples)} examples")
    return examples


LOADERS = {
    "panel_primary": load_panel_primary,
    "swiid_summary": load_swiid_summary,
    "vdem_polyarchy": load_vdem_polyarchy,
    "maddison_gdppc": load_maddison,
    "wb_tot": load_wb_tot,
}


def main() -> None:
    datasets = []
    for name, loader in LOADERS.items():
        try:
            examples = loader()
            if examples:
                datasets.append({"dataset": name, "examples": examples})
                logger.info(f"  ✓ {name}: {len(examples)} examples")
        except Exception as e:
            logger.error(f"  ✗ {name}: {e}")

    total = sum(len(d["examples"]) for d in datasets)
    logger.info(f"\nTotal: {len(datasets)} datasets, {total} examples")

    out = {"datasets": datasets}
    out_path = WS / "full_data_out.json"
    out_path.write_text(json.dumps(out))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved full_data_out.json ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
