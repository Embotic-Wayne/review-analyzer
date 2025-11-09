import re
import pandas as pd
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def normalize_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def guess_columns(df: pd.DataFrame):
    # try common names
    text_cols = [c for c in df.columns if c.lower() in ["reviewtext","text","review","body","content","review_text"]]
    rating_cols = [c for c in df.columns if c.lower() in ["score","rating","stars"]]
    pid_cols = [c for c in df.columns if c.lower() in ["productid","product_id","asin"]]
    ptitle_cols = [c for c in df.columns if c.lower() in ["product_title","title","productname","product_name"]]

    text_col = text_cols[0] if text_cols else df.columns[0]
    rating_col = rating_cols[0] if rating_cols else None
    pid_col = pid_cols[0] if pid_cols else None
    ptitle_col = ptitle_cols[0] if ptitle_cols else None
    return text_col, rating_col, pid_col, ptitle_col
