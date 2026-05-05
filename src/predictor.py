from __future__ import annotations  # para future-compatible ang type hints

import os  # para file paths
from collections import Counter  # (wala gigamit diri, pwede pa ni nimo magamit later)
from typing import List, Tuple  # type hinting

import joblib  # para save/load model
import numpy as np  # numerical operations
import pandas as pd  # dataframe handling
from sklearn.feature_extraction.text import TfidfVectorizer  # convert text → numbers
from sklearn.linear_model import LogisticRegression  # ML model
from sklearn.pipeline import Pipeline  # para combine steps

# import sa imong sariling files
from .bootstrap_data import get_bootstrap_dataset  # training data
from .nlp_utils import POSITIVE_HINTS, NEGATIVE_HINTS, NEUTRAL_HINTS, normalize_text, preprocess_text

# path kung asa i-save ang trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sentiment_pipeline.joblib")

# =========================
# BUILD PIPELINE
# =========================
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess_text,  # custom preprocessing (clean text)
            ngram_range=(1, 2),  # unigram + bigram
            min_df=1,  # minimum frequency
            max_features=6000,  # limit features
            sublinear_tf=True  # scaling
        )),
        ("clf", LogisticRegression(
            max_iter=1500,  # para dili mag convergence error
            class_weight="balanced",  # balance classes
            random_state=42
        )),
    ])

# =========================
# TRAIN OR LOAD MODEL
# =========================
def train_or_load_pipeline(force_retrain: bool = False) -> Pipeline:
    
    # kung naa nay existing model ug dili ka mag retrain → load lang
    if os.path.exists(MODEL_PATH) and not force_retrain:
        return joblib.load(MODEL_PATH)

    # kuha dataset
    rows = get_bootstrap_dataset()

    # separate text ug label
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    # build pipeline
    pipe = build_pipeline()

    # train model
    pipe.fit(texts, labels)

    # create folder kung wala pa
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # save model
    joblib.dump(pipe, MODEL_PATH)

    return pipe

# =========================
# RULE-BASED ADJUSTMENT
# =========================
def rule_adjustment(text: str, proba: np.ndarray, classes: List[str]) -> np.ndarray:
    
    # normalize text (lowercase, remove noise)
    normalized = normalize_text(text)

    # tokenize
    tokens = normalized.split()
    token_set = set(tokens)

    # count hint words
    pos_hits = len(token_set & POSITIVE_HINTS)
    neg_hits = len(token_set & NEGATIVE_HINTS)
    neu_hits = len(token_set & NEUTRAL_HINTS)

    # copy original probabilities gikan sa ML model
    adjusted = proba.copy()

    # mapping: label → index
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # boost scores base sa detected words
    if "positive" in class_to_idx:
        adjusted[class_to_idx["positive"]] += 0.03 * pos_hits

    if "negative" in class_to_idx:
        adjusted[class_to_idx["negative"]] += 0.03 * neg_hits

    if "neutral" in class_to_idx:
        adjusted[class_to_idx["neutral"]] += 0.02 * neu_hits

    # =========================
    # MIXED SENTIMENT (important kaayo ni )
    # =========================
    contrast_words = {"but", "however", "though", "although","yet","still"}

    # example: "good BUT bad"
    if any(w in normalized.split() for w in contrast_words) and pos_hits > 0 and neg_hits > 0:
        adjusted[class_to_idx["neutral"]] += 0.15  # push ngadto sa neutral

    # =========================
    # SHORT TEXT HANDLING
    # =========================
    if len(tokens) <= 3 and neu_hits > 0:
        adjusted[class_to_idx["neutral"]] += 0.10

    # para walay zero values (important sa normalization)
    adjusted = np.maximum(adjusted, 1e-9)

    # normalize → total = 1
    adjusted = adjusted / adjusted.sum()

    return adjusted

# =========================
# PREDICT DATAFRAME
# =========================
def predict_dataframe(df: pd.DataFrame, text_col: str):

    # load or train model
    pipe = train_or_load_pipeline()

    # kuha texts
    texts = df[text_col].fillna("").astype(str).tolist()

    # base prediction gikan ML
    base_proba = pipe.predict_proba(texts)

    # list of class labels
    classes = list(pipe.classes_)

    labels, confs = [], []
    pos_scores, neg_scores, neu_scores = [], [], []

    # loop bawat row
    for text, row_proba in zip(texts, base_proba):

        # apply hybrid rule adjustment
        adj = rule_adjustment(text, row_proba, classes)

        # get highest score
        idx = int(np.argmax(adj))
        label = classes[idx]
        conf = float(adj[idx])

        # store results
        labels.append(label)
        confs.append(conf)

        # store detailed scores
        pos_scores.append(float(adj[classes.index("positive")]))
        neg_scores.append(float(adj[classes.index("negative")]))
        neu_scores.append(float(adj[classes.index("neutral")]))

    # create result dataframe
    result = df.copy()
    result["predicted_sentiment"] = labels
    result["confidence"] = confs
    result["positive_score"] = pos_scores
    result["negative_score"] = neg_scores
    result["neutral_score"] = neu_scores

    return result

# =========================
# CONFIDENCE LEVEL
# =========================
def confidence_band(score: float) -> str:
    
    # high confidence
    if score >= 0.75:
        return "High"

    # medium
    if score >= 0.55:
        return "Medium"

    # low
    return "Low"