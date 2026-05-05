# AI Sentiment Studio — ML + NLP Edition

A professional Streamlit dashboard for sentiment analysis using **machine learning** and **natural language processing**.

## What it does
- Upload an Excel or CSV file with **text only**
- Auto-detect the text column
- Preprocess text using NLP
- Predict **positive**, **negative**, or **neutral** using a local ML model
- Show sentiment distribution charts
- Show top positive and negative keywords
- Download the prediction report

## Tech stack
- Python
- Streamlit
- scikit-learn
- pandas
- matplotlib
- openpyxl
- joblib

## Project structure
```text
ai_sentiment_studio_ml_nlp/
├── app/
│   └── app.py
├── src/
│   ├── bootstrap_data.py
│   ├── nlp_utils.py
│   ├── predictor.py
│   └── keyword_utils.py
├── models/
├── data/
├── assets/
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

## Input format
Upload a CSV or Excel file with one text column such as:
- review
- text
- content
- comment
- feedback
- tweet
- message

The app auto-detects the most likely text column.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Notes
- This version uses **local ML prediction only**
- No OpenAI calls are used for prediction
- The bundled training corpus is internal and used automatically to fit the model
- Trained artifacts are saved automatically with `joblib` inside the `models/` folder

## Suggested GitHub description
**Built a professional sentiment analysis dashboard using Python, Streamlit, NLP, and machine learning. Implemented TF-IDF text vectorization, Logistic Regression classification, automatic text-column detection, keyword analysis, and downloadable prediction reports.**
