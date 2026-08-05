"""
Stage 1: Data Ingestion
------------------------
Reads the raw Boston Housing CSV (HousingData.csv) from its source
location and copies it into the pipeline's raw data folder.

Input:
    data/external/HousingData.csv   (place your downloaded CSV here)
Output:
    data/raw/data.csv
"""

import os
import yaml
import pandas as pd


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(source_path: str) -> pd.DataFrame:
    df = pd.read_csv(source_path)
    print(f"[data_ingestion] Loaded raw data from {source_path} (shape={df.shape})")
    return df


def save_raw_data(df: pd.DataFrame, out_dir: str = "data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_ingestion] Saved raw data -> {out_path} (shape={df.shape})")


def main():
    params = load_params()["data_ingestion"]
    df = load_data(params["raw_data_path"])
    save_raw_data(df)


if __name__ == "__main__":
    main()
