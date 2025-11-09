import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Smart Review Analyzer", layout="wide")

DATA_FILE = Path("data/processed/reviews_with_sentiment.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()
st.title("🛒 Smart Product Review Analyzer")

# Filters
product_col = "product_title" if "product_title" in df.columns else ("product_id" if "product_id" in df.columns else None)
if product_col:
    products = ["(All)"] + sorted(df[product_col].dropna().unique().tolist())
    selected = st.sidebar.selectbox("Product", products)
    if selected != "(All)":
        df = df[df[product_col] == selected]

# Sentiment distribution
sent_counts = df["sentiment_pred"].value_counts().reset_index()
sent_counts.columns = ["sentiment", "count"]
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Sentiment Distribution")
    fig = px.bar(sent_counts, x="sentiment", y="count", text="count")
    st.plotly_chart(fig, use_container_width=True)

# Top topics
if "topic" in df.columns:
    top_topics = df["topic"].value_counts().head(10).reset_index()
    top_topics.columns = ["topic", "count"]
    with col2:
        st.subheader("Top Topics")
        fig2 = px.bar(top_topics, x="topic", y="count", text="count")
        st.plotly_chart(fig2, use_container_width=True)

# Keyword search & examples
st.subheader("Example Reviews")
query = st.text_input("Filter by keyword", "")
sub = df.copy()
if query:
    sub = sub[sub["review_text_clean"].str.contains(query.lower(), na=False)]

# Show a few examples per sentiment
for s in ["negative", "neutral", "positive"]:
    st.markdown(f"**{s.title()} samples**")
    view = sub[sub["sentiment_pred"] == s].head(5)
    for _, r in view.iterrows():
        st.write(f"- {r.get('review_text', r.get('review_text_clean',''))[:300]}…")
