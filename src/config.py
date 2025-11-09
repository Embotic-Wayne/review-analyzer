from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")

RAW_FILE = RAW_DIR / "reviews.csv"
CLEAN_FILE = PROCESSED_DIR / "reviews_clean.csv"
SENTIMENT_FILE = PROCESSED_DIR / "reviews_with_sentiment.csv"
TOPICS_FILE = PROCESSED_DIR / "topics.csv"
