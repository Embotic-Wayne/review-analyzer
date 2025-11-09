import pandas as pd
from pathlib import Path
from src.config import RAW_FILE, PROCESSED_DIR, CLEAN_FILE
from src.utils import normalize_text, guess_columns

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    assert RAW_FILE.exists(), f"Missing raw file at {RAW_FILE}"
    df = pd.read_csv(RAW_FILE)
    text_col, rating_col, pid_col, ptitle_col = guess_columns(df)

    df = df.rename(columns={
        text_col: "review_text",
        rating_col if rating_col else text_col: "rating",   # if no rating, fill later
    })
    if pid_col: df = df.rename(columns={pid_col: "product_id"})
    if ptitle_col: df = df.rename(columns={ptitle_col: "product_title"})

    if "rating" not in df.columns:
        df["rating"] = None

    df["review_text_clean"] = df["review_text"].apply(normalize_text)
    df = df.dropna(subset=["review_text_clean"]).query("review_text_clean.str.len()>0")

    df.to_csv(CLEAN_FILE, index=False)
    print(f"Saved cleaned data to {CLEAN_FILE} with {len(df)} rows.")

if __name__ == "__main__":
    main()
