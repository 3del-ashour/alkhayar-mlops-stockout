import pandas as pd
from pathlib import Path


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def load_sample(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / name)
