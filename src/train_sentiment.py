# src/train_sentiment.py
import os
from pathlib import Path
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download

from src.config import CLEAN_FILE, SENTIMENT_FILE
from src.config import PROCESSED_DIR  # ensure PROCESSED_DIR exists

def star_to_label(star):
    try:
        s = float(star)
        if s >= 4: return "positive"
        if s <= 2: return "negative"
        return "neutral"
    except:
        return None

def main():
    # Ensure directories
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(CLEAN_FILE).exists():
        raise FileNotFoundError(f"Missing cleaned data at {CLEAN_FILE}. Run: python -m src.prepare_data")

    print(f"[info] Loading {CLEAN_FILE} ...")
    df = pd.read_csv(CLEAN_FILE)

    # Ensure NLTK resources
    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        print("[info] Downloading missing NLTK resources...")
        nltk_download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()

    print(f"[info] Scoring {len(df):,} reviews with VADER (this can take a few minutes on large files)...")
    df["vader_compound"] = df["review_text_clean"].astype(str).map(lambda t: sia.polarity_scores(t)["compound"])
    df["sentiment_pred"] = df["vader_compound"].map(lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral"))

    # Optional agreement vs stars
    if "rating" in df.columns and df["rating"].notna().any():
        df["sentiment_true"] = df["rating"].map(star_to_label)
        labeled = df.dropna(subset=["sentiment_true"])
        if len(labeled):
            acc = (labeled["sentiment_true"] == labeled["sentiment_pred"]).mean()
            print(f"[info] VADER agreement vs stars (rough): {acc:.3f} on {len(labeled):,} labeled rows")

    # Save
    out_path = Path(SENTIMENT_FILE)
    df.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024*1024)
    print(f"[ok] Saved with sentiment to {out_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
