import pandas as pd
from sklearn.metrics import classification_report
from src.config import SENTIMENT_FILE

def main():
    df = pd.read_csv(SENTIMENT_FILE)
    if "rating" not in df.columns:
        print("No ground truth ratings available.")
        return

    def star_to_label(s):
        try:
            s = float(s)
            if s >= 4: return "positive"
            if s <= 2: return "negative"
            return "neutral"
        except:
            return None

    labeled = df.dropna(subset=["rating"]).copy()
    labeled["sentiment_true"] = labeled["rating"].apply(star_to_label)
    labeled = labeled.dropna(subset=["sentiment_true"])
    print(classification_report(labeled["sentiment_true"], labeled["sentiment_pred"]))
    # You can compute per-product summaries here, too.

if __name__ == "__main__":
    main()
