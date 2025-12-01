# 🧠 Amazon Product Review Analyzer  
> NLP pipeline for sentiment analysis and topic discovery on Amazon product reviews  

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Processing-orange.svg)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-lightblue.svg)](https://plotly.com/)

---

## 📘 Overview
The **Amazon Product Review Analyzer** is a Natural Language Processing (NLP) project that extracts insights from thousands of Amazon reviews.  
It combines **BERT** for sentiment classification with **LDA** and **BERTopic** for topic modeling to uncover recurring complaints and praise themes.  
Results are visualized through interactive **Plotly** charts for easy interpretation.

---

## 🚀 Features
- 🔍 **Sentiment Analysis:** Fine-tuned BERT model (via HuggingFace Transformers) to classify reviews as positive, neutral, or negative.  
- 🧩 **Topic Modeling:** LDA and BERTopic used to uncover common complaint categories and product feedback themes.  
- 🧹 **Data Processing:** Cleaned and preprocessed 10k+ Kaggle Amazon reviews using pandas.  
- 📊 **Visualization:** Interactive dashboards built with Plotly for sentiment and topic exploration.  
- ⚙️ **Pipeline Design:** Modular Python scripts for data prep, training, evaluation, and topic discovery.

---

## 🧰 Tech Stack
| Technology | Purpose |
|-------------|----------|
| **Python** | Core language for data processing, modeling, and visualization |
| **pandas** | Data cleaning, manipulation, and preprocessing |
| **HuggingFace Transformers** | BERT model fine-tuning for sentiment classification |
| **scikit-learn (LDA)** | Classical topic modeling using Latent Dirichlet Allocation |
| **BERTopic** | Transformer-based topic discovery with better contextual clusters |
| **Plotly** | Interactive visualizations and charts for results presentation |

---

## 📊 Results
- **92% F1 score** on held-out test data using fine-tuned BERT for sentiment classification.  
- **BERTopic/LDA** extracted clear complaint categories such as *packaging issues*, *delivery delays*, and *product defects*.  
- Visual insights enabled clear understanding of customer pain points for potential product improvements.

---

## 🧪 How It Works
1. **Data Preparation:**  
   - Load dataset (`reviews.csv`) using pandas.  
   - Clean and normalize text (remove punctuation, stopwords, URLs).  

2. **Sentiment Analysis (BERT):**  
   - Fine-tune a pre-trained BERT model from HuggingFace on labeled review data.  
   - Evaluate using F1, precision, and recall metrics.  

3. **Topic Modeling (LDA & BERTopic):**  
   - Apply LDA for initial topic discovery.  
   - Use BERTopic for transformer-based, semantically richer topics.  

4. **Visualization:**  
   - Generate Plotly charts for sentiment distribution and top complaint themes.

## ⚡ Quickstart

### 
1️⃣  Create a virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Add the dataset
Place your Amazon reviews CSV at:
data/raw/reviews.csv
(Any CSV with a Text or review_text column works.)

3️⃣ Preprocess the data
python -m src.prepare_data

4️⃣ Run sentiment analysis (BERT/VADER)
python -m src.train_sentiment

5️⃣ Run topic modeling (LDA or BERTopic)
python -m src.topic_model

6️⃣Visualize results (optional Streamlit dashboard)
streamlit run app/streamlit_app.py
Then go to http://localhost:8501 in your browser.
