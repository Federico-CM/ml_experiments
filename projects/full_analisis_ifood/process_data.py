from pathlib import Path
import pandas as pd
import numpy as np


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path).copy()

    # Drop uninformative columns
    df = df.drop(columns=["Z_CostContact", "Z_Revenue", "ID"], errors="ignore")

    # Fix missing income with median
    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(df["Income"].median())

    # Handle implausible birth years
    if "Year_Birth" in df.columns:
        df.loc[df["Year_Birth"].between(1893, 1900), "Year_Birth"] = pd.NA
        df = df.dropna(subset=["Year_Birth"])

    # Fix invalid marital status values
    if "Marital_Status" in df.columns:
        invalid_status = ["Absurd", "YOLO"]
        df.loc[df["Marital_Status"].isin(invalid_status), "Marital_Status"] = pd.NA
        df = df.dropna(subset=["Marital_Status"])

    # Parse customer date
    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")

    # Log-transform income
    if "Income" in df.columns:
        df["Income_log"] = np.log1p(df["Income"])

    return df
