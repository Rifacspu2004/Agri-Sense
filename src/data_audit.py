"""
Data Audit Module for Kaggle Crop Yield Prediction Dataset.
Inspects columns: Area, Item, Year, rainfall, pesticides, temperature, and yield target.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


REQUIRED_COLUMNS = [
    "Area",
    "Item",
    "Year",
    "hg/ha_yield",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]


def run_data_audit(csv_path: str = "data/raw/yield_df.csv", output_dir: str = "artifacts") -> Dict[str, Any]:
    """
    Audits the input dataset to ensure schema compliance, statistical validity,
    and absence of critical missing values.
    """
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(file_path)

    # 1. Column Verification
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # 2. Schema and Missingness Check
    total_rows = int(len(df))
    null_counts = {col: int(df[col].isna().sum()) for col in REQUIRED_COLUMNS}
    dtypes = {col: str(df[col].dtype) for col in REQUIRED_COLUMNS}

    # 3. Categorical distribution
    unique_areas: List[str] = sorted(df["Area"].dropna().unique().tolist())
    unique_items: List[str] = sorted(df["Item"].dropna().unique().tolist())

    # 4. Numerical summary statistics
    num_cols = ["Year", "hg/ha_yield", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    num_stats: Dict[str, Dict[str, float]] = {}
    for col in num_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        num_stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            "nulls": int(series.isna().sum())
        }

    audit_summary: Dict[str, Any] = {
        "status": "PASSED" if not missing_cols and all(v == 0 for v in null_counts.values()) else "WARNING",
        "dataset_path": str(file_path),
        "total_records": total_rows,
        "unique_areas_count": len(unique_areas),
        "unique_crops_count": len(unique_items),
        "areas_sample": unique_areas[:10],
        "crops_sample": unique_items,
        "null_counts": null_counts,
        "column_dtypes": dtypes,
        "numeric_statistics": num_stats,
        "target_yield_unit": "hg/ha (10,000 hg/ha = 1 tonne/ha)"
    }

    # Save to artifacts directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "audit_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("=== Data Audit Summary ===")
    print(f"Status: {audit_summary['status']}")
    print(f"Total Rows: {total_rows}")
    print(f"Distinct Countries/Areas: {len(unique_areas)}")
    print(f"Distinct Crops (Items): {len(unique_items)}")
    print(f"Yield Range (hg/ha): {num_stats['hg/ha_yield']['min']} -> {num_stats['hg/ha_yield']['max']}")
    print(f"Audit report saved to: {report_file}")

    return audit_summary


if __name__ == "__main__":
    run_data_audit()
