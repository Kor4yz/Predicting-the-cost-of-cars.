from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def load_raw(filename: str) -> pd.DataFrame:
    path = RAW / filename
    return pd.read_csv(path)

def save_processed(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(PROC / filename, index=False)
