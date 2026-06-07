#!/usr/bin/env python3
"""Build Democratic Resilience Panel: SWIID + V-Dem + Maddison + WB indicators."""

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


def load_swiid() -> pd.DataFrame:
    df = pd.read_csv(DS / "swiid_summary.csv")
    df = df[df["year"] >= 1990][["country", "year", "gini_disp", "gini_disp_se", "gini_mkt", "gini_mkt_se"]]
    logger.info(f"SWIID: {len(df)} rows, {df['country'].nunique()} countries")
    return df


def load_owid_democracy() -> pd.DataFrame:
    """V-Dem polyarchy and libdem from OWID grapher."""
    poly = pd.read_csv(DS / "owid_vdem_polyarchy.csv")
    poly = poly.rename(columns={"Entity": "country", "Code": "iso3", "Year": "year", "Electoral democracy index": "v2x_polyarchy"})
    poly = poly[poly["year"] >= 1990][["country", "iso3", "year", "v2x_polyarchy"]]

    libdem = pd.read_csv(DS / "owid_vdem_libdem.csv")
    libdem = libdem.rename(columns={"Entity": "country", "Code": "iso3", "Year": "year", "Liberal democracy index": "v2x_libdem"})
    libdem = libdem[libdem["year"] >= 1990][["iso3", "year", "v2x_libdem"]]

    df = poly.merge(libdem, on=["iso3", "year"], how="outer")
    # Drop aggregate regions (no ISO3 code)
    df = df[df["iso3"].notna() & (df["iso3"].str.len() == 3)]
    logger.info(f"V-Dem OWID: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df


def load_maddison() -> pd.DataFrame:
    """Maddison Project Database GDP per capita (2011 PPP USD)."""
    path = OWID_TB / "full_garden_ggdc_2024-04-26_maddison_project_database_maddison_project_database.json"
    raw = json.loads(path.read_text())
    df = pd.DataFrame(raw)
    df = df[df["year"] >= 1990][["country", "year", "gdp_per_capita"]]
    df = df.rename(columns={"gdp_per_capita": "gdppc_maddison"})
    df = df[df["gdppc_maddison"].notna()]
    logger.info(f"Maddison: {len(df)} rows, {df['country'].nunique()} countries")
    return df


def parse_wb_json(path: Path, value_col: str) -> pd.DataFrame:
    """Parse World Bank API response JSON into (iso3, year, value) DataFrame."""
    raw = json.loads(path.read_text())
    rows = raw[1] if isinstance(raw, list) and len(raw) > 1 and raw[1] else []
    records = []
    for r in rows:
        iso3 = r.get("countryiso3code", "")
        year = r.get("date")
        val = r.get("value")
        if iso3 and len(iso3) == 3 and year and val is not None:
            try:
                records.append({"iso3": iso3, "year": int(year), value_col: float(val)})
            except (ValueError, TypeError):
                pass
    df = pd.DataFrame(records)
    logger.info(f"WB {value_col}: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df


def load_world_bank() -> pd.DataFrame:
    gdp_growth = parse_wb_json(DS / "wb_gdp_growth_raw.json", "gdp_growth")
    gdppc_ppp = parse_wb_json(DS / "wb_gdppc_ppp_raw.json", "gdppc_ppp_wb")
    tot = parse_wb_json(DS / "wb_tot_raw.json", "terms_of_trade")
    df = gdp_growth.merge(gdppc_ppp, on=["iso3", "year"], how="outer")
    df = df.merge(tot, on=["iso3", "year"], how="outer")
    df = df[df["year"] >= 1990]
    return df


def build_iso3_crosswalk(vdem: pd.DataFrame) -> dict[str, str]:
    """Build country name → ISO3 map from V-Dem OWID (which has both)."""
    return dict(zip(vdem["country"].str.strip(), vdem["iso3"]))


def name_to_iso3(series: pd.Series, crosswalk: dict[str, str]) -> pd.Series:
    """Map country names to ISO3 using crosswalk."""
    manual = {
        "Czechia": "CZE",
        "Czech Republic": "CZE",
        "Slovak Republic": "SVK",
        "Slovakia": "SVK",
        "South Korea": "KOR",
        "Korea": "KOR",
        "Republic of Korea": "KOR",
        "North Macedonia": "MKD",
        "Macedonia": "MKD",
        "Eswatini": "SWZ",
        "Swaziland": "SWZ",
        "Congo": "COG",
        "DR Congo": "COD",
        "Democratic Republic of Congo": "COD",
        "Ivory Coast": "CIV",
        "Cote d'Ivoire": "CIV",
        "Cape Verde": "CPV",
        "Timor-Leste": "TLS",
        "East Timor": "TLS",
        "Palestinian Territories": "PSE",
        "Palestine": "PSE",
        "Bosnia & Herzegovina": "BIH",
        "Bosnia and Herzegovina": "BIH",
        "Trinidad & Tobago": "TTO",
        "Trinidad and Tobago": "TTO",
        "Kyrgyz Republic": "KGZ",
        "Kyrgyzstan": "KGZ",
        "Lao PDR": "LAO",
        "Laos": "LAO",
        "São Tomé and Príncipe": "STP",
        "Sao Tome and Principe": "STP",
        "Gambia": "GMB",
        "The Gambia": "GMB",
    }
    combined = {**crosswalk, **manual}
    return series.str.strip().map(combined)


def max_consecutive_post1990(years: pd.Series) -> int:
    """Return length of the longest consecutive run in sorted year list."""
    yrs = sorted(years.dropna().unique().astype(int))
    if not yrs:
        return 0
    max_run = run = 1
    for a, b in zip(yrs, yrs[1:]):
        if b == a + 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run


def build_panel(min_consecutive: int = 15) -> pd.DataFrame:
    swiid = load_swiid()
    vdem = load_owid_democracy()
    maddison = load_maddison()
    wb = load_world_bank()

    crosswalk = build_iso3_crosswalk(vdem)

    # Add iso3 to SWIID and Maddison
    swiid["iso3"] = name_to_iso3(swiid["country"], crosswalk)
    maddison["iso3"] = name_to_iso3(maddison["country"], crosswalk)

    # Merge: start from SWIID (our outcome), join everything on (iso3, year)
    panel = swiid.dropna(subset=["iso3"]).merge(
        vdem[["iso3", "year", "v2x_polyarchy", "v2x_libdem"]],
        on=["iso3", "year"], how="left"
    ).merge(
        maddison[["iso3", "year", "gdppc_maddison"]],
        on=["iso3", "year"], how="left"
    ).merge(
        wb, on=["iso3", "year"], how="left"
    )

    # Log GDP per capita (use Maddison, fall back to WB PPP)
    panel["ln_gdppc"] = panel["gdppc_maddison"].where(
        panel["gdppc_maddison"].notna(), panel["gdppc_ppp_wb"]
    ).apply(lambda x: x if pd.isna(x) else max(x, 1)).apply(
        lambda x: pd.NA if pd.isna(x) else float(__import__("math").log(x))
    )

    logger.info(f"Merged panel: {len(panel)} rows, {panel['iso3'].nunique()} countries")

    # Filter to T>=15 consecutive post-1990 observations with gini_disp
    key_vars = ["gini_disp", "v2x_polyarchy"]
    panel_core = panel.dropna(subset=key_vars)
    consec = panel_core.groupby("iso3")["year"].apply(max_consecutive_post1990)
    keep = consec[consec >= min_consecutive].index
    panel_full = panel[panel["iso3"].isin(keep)].copy()
    logger.info(f"After T>={min_consecutive} filter: {len(panel_full)} rows, {panel_full['iso3'].nunique()} countries")

    return panel_full


def build_primary(panel: pd.DataFrame) -> pd.DataFrame:
    """Balanced panel keeping only iso3 with >=15 non-missing rows on all primary vars."""
    primary_vars = ["gini_disp", "v2x_polyarchy", "ln_gdppc", "gdp_growth"]
    df = panel.dropna(subset=primary_vars)
    counts = df.groupby("iso3").size()
    keep = counts[counts >= 15].index
    df = df[df["iso3"].isin(keep)].copy()
    logger.info(f"Primary panel: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df


def make_diagnostics(panel: pd.DataFrame, primary: pd.DataFrame) -> dict:
    def coverage(df: pd.DataFrame) -> dict:
        return {
            col: {
                "n_nonmissing": int(df[col].notna().sum()),
                "pct_coverage": round(df[col].notna().mean() * 100, 1),
            }
            for col in df.columns if df[col].dtype in ["float64", "int64", "Float64"]
        }

    return {
        "panel_full": {
            "n_rows": len(panel),
            "n_countries": int(panel["iso3"].nunique()),
            "year_range": [int(panel["year"].min()), int(panel["year"].max())],
            "variable_coverage": coverage(panel),
        },
        "panel_primary": {
            "n_rows": len(primary),
            "n_countries": int(primary["iso3"].nunique()),
            "year_range": [int(primary["year"].min()), int(primary["year"].max())],
        },
    }


def rows_to_examples(df: pd.DataFrame, outcome: str = "gini_disp") -> list[dict]:
    """Convert panel rows to {input, output} pairs for the schema."""
    predictor_cols = [c for c in df.columns if c != outcome]
    examples = []
    for _, row in df.head(3).iterrows():
        predictors = {c: (None if pd.isna(row[c]) else row[c]) for c in predictor_cols}
        out_val = row.get(outcome)
        examples.append({
            "input": json.dumps(predictors, default=str),
            "output": str(round(out_val, 4)) if pd.notna(out_val) else "",
        })
    return examples


def make_data_out(panel: pd.DataFrame, primary: pd.DataFrame, diag: dict) -> dict:
    return {
        "metadata": {
            "description": "Democratic Resilience Panel: SWIID + V-Dem + Maddison + World Bank",
            "panel_full_file": "panel_full.parquet",
            "panel_primary_file": "panel_primary.parquet",
        },
        "datasets": [
            {
                "dataset": "democratic_resilience_panel_full",
                "examples": rows_to_examples(panel),
            },
            {
                "dataset": "democratic_resilience_panel_primary",
                "examples": rows_to_examples(primary),
            },
        ]
    }


def main() -> None:
    panel = build_panel(min_consecutive=15)
    primary = build_primary(panel)
    diag = make_diagnostics(panel, primary)

    # Save parquet
    panel.to_parquet(WS / "panel_full.parquet", index=False)
    primary.to_parquet(WS / "panel_primary.parquet", index=False)
    logger.info("Saved panel_full.parquet and panel_primary.parquet")

    # Save panel_full.json
    panel_json = panel.to_json(orient="records")
    (WS / "panel_full.json").write_text(panel_json)
    logger.info(f"Saved panel_full.json ({(WS / 'panel_full.json').stat().st_size / 1e6:.1f}MB)")

    # Save diagnostics
    (WS / "diagnostics_summary.json").write_text(json.dumps(diag, indent=2))
    logger.info("Saved diagnostics_summary.json")

    # Save data_out.json
    data_out = make_data_out(panel, primary, diag)
    (WS / "data_out.json").write_text(json.dumps(data_out, indent=2))
    logger.info("Saved data_out.json")

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Panel full: {diag['panel_full']['n_rows']} rows, {diag['panel_full']['n_countries']} countries")
    logger.info(f"Panel primary: {diag['panel_primary']['n_rows']} rows, {diag['panel_primary']['n_countries']} countries")


if __name__ == "__main__":
    main()
