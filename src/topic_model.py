# src/topic_model.py (quick mode & robust)
import os, sys
import pandas as pd
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from src.config import SENTIMENT_FILE, CLEAN_FILE, TOPICS_FILE

# ---- TUNABLES (safe, fast defaults) ----
N_TOPICS = int(os.getenv("N_TOPICS", "8"))
N_WORDS  = int(os.getenv("N_WORDS", "10"))
SAMPLE_N = int(os.getenv("TOPIC_SAMPLE_N", "100000"))  # set smaller/bigger via env var
MIN_DF   = int(os.getenv("TOPIC_MIN_DF", "5"))
MAX_FEAT = int(os.getenv("TOPIC_MAX_FEATURES", "50000"))
MAX_DF   = float(os.getenv("TOPIC_MAX_DF", "0.90"))

def load_input():
    if Path(SENTIMENT_FILE).exists():
        print(f"[info] Loading {SENTIMENT_FILE} ...")
        df = pd.read_csv(SENTIMENT_FILE)
    elif Path(CLEAN_FILE).exists():
        print(f"[info] {SENTIMENT_FILE} not found. Using {CLEAN_FILE} instead.")
        df = pd.read_csv(CLEAN_FILE)
    else:
        sys.exit(f"[error] Neither {SENTIMENT_FILE} nor {CLEAN_FILE} exists. Run: python -m src.prepare_data")
    if "review_text_clean" not in df.columns:
        sys.exit("[error] 'review_text_clean' missing. Re-run: python -m src.prepare_data")
    return df

def main():
    df = load_input()
    n_docs_total = len(df)
    print(f"[info] Loaded {n_docs_total:,} rows")

    # sample to speed up
    if n_docs_total > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=42)
        print(f"[info] Sampled down to {len(df):,} rows for quick modeling (set TOPIC_SAMPLE_N to change)")

    texts = df["review_text_clean"].astype(str).tolist()

    # Adjust min_df for very small samples
    eff_min_df = 1 if len(texts) < 200 else MIN_DF

    print(f"[info] Vectorizing (min_df={eff_min_df}, max_df={MAX_DF}, max_features={MAX_FEAT}) ...")
    vect = CountVectorizer(min_df=eff_min_df, max_df=MAX_DF, max_features=MAX_FEAT, stop_words="english")
    X = vect.fit_transform(texts)
    if X.shape[1] == 0:
        sys.exit("[error] Empty vocabulary. Add more reviews or lower MIN_DF / raise MAX_FEATURES.")

    print(f"[info] Fitting LDA (n_topics={N_TOPICS}, online solver) on shape={X.shape} ...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        learning_method="online",  # faster on large corpora
        max_iter=10,
        batch_size=2048,
        evaluate_every=0,
        n_jobs=-1,
        random_state=42,
    )
    W = lda.fit_transform(X)   # doc-topic
    H = lda.components_        # topic-word
    print("[info] LDA done.")

    vocab = pd.Series(vect.get_feature_names_out())
    topic_rows = []
    for k in range(N_TOPICS):
        top_idx = H[k].argsort()[-N_WORDS:][::-1]
        words = vocab.iloc[top_idx].tolist()
        topic_rows.append({"topic": k, "top_words": ", ".join(words)})

    # Persist
    Path(TOPICS_FILE).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(topic_rows).to_csv(TOPICS_FILE, index=False)
    df["topic"] = W.argmax(axis=1)

    # Save back to whichever file is powering the app
    out_path = Path(SENTIMENT_FILE) if Path(SENTIMENT_FILE).exists() else Path(CLEAN_FILE)
    df.to_csv(out_path, index=False)

    print(f"[ok] Saved topics to {TOPICS_FILE}")
    print(f"[ok] Updated topic assignments in {out_path}")

if __name__ == "__main__":
    main()
