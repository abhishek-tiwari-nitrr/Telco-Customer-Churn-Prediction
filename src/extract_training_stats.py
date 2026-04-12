import pandas as pd
import json
from pathlib import Path
from config import TRAINING_DATA_PATH, NUMERIC_COLUMNS, DATA_DIR


def get_stats():
    df = pd.read_csv(TRAINING_DATA_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    stats = {}
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            stats[col] = {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "missing_rate": round(float(df[col].isna().mean()), 4),
            }

    churn_map = {"Yes": 1, "No": 0}
    stats["churn_rate"] = round(float(df["churn"].map(churn_map).mean()), 4)

    cat_cols = [
        "contract",
        "internetservice",
        "paymentmethod",
        "partner",
        "dependents",
        "phoneservice",
    ]
    for col in cat_cols:
        if col in df.columns:
            dist = {
                k: round(v, 4)
                for k, v in df[col].value_counts(normalize=True).to_dict().items()
            }
            stats[f"{col}_dist"] = dist

    out_path = Path(DATA_DIR / "training_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=4))


if __name__ == "__main__":
    get_stats()
