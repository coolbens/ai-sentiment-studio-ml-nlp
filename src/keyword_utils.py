
from __future__ import annotations  # Para future compatibility sa type hints (mas limpyo ang syntax)

from collections import Counter  # Gigamit para ihap (count) ang frequency sa words
from typing import Dict, List, Tuple  # Type hints para klaro ang expected data types

import pandas as pd  # Pandas para data manipulation (DataFrame)

from .nlp_utils import preprocess_text, POSITIVE_HINTS, NEGATIVE_HINTS  
# preprocess_text = cleaning sa text (lowercase, remove punctuation, etc.)
# POSITIVE_HINTS / NEGATIVE_HINTS = predefined keyword sets

def extract_top_keywords(
    df: pd.DataFrame, 
    text_col: str, 
    sentiment_col: str, 
    target_sentiment: str, 
    top_n: int = 12
):
    # STEP 1: I-filter ang DataFrame base sa target sentiment (positive / negative / neutral)
    subset = df[df[sentiment_col] == target_sentiment]

    # STEP 2: Mag-create og counter para ihap ang frequency sa words
    counter = Counter()

    # STEP 3: Loop sa tanan text sa selected subset
    for text in subset[text_col].fillna("").astype(str):
        # fillna("") = para walay None values
        # astype(str) = siguraduhon nga string tanan

        # preprocess_text = cleaning + normalization sa text
        for tok in preprocess_text(text).split():
            # split() = himuon list sa words/tokens

            # STEP 4: Ignore short words (less than 3 characters)
            if len(tok) >= 3:
                counter[tok] += 1  # add count sa word

    # STEP 5: Filtering logic based on sentiment
    if target_sentiment == "positive":
        # I-keep lang:
        # - words nga naa sa POSITIVE_HINTS OR
        # - words nga gi-mention at least 2 times
        filtered = {
            k: v for k, v in counter.items() 
            if k in POSITIVE_HINTS or v >= 2
        }

    elif target_sentiment == "negative":
        # Same logic pero gamit NEGATIVE_HINTS
        filtered = {
            k: v for k, v in counter.items() 
            if k in NEGATIVE_HINTS or v >= 2
        }

    else:
        # Kung neutral or uban pa, walay filtering
        filtered = dict(counter)

    # STEP 6: Sort keywords
    # - Highest frequency first (-x[1])
    # - Alphabetical if tie (x[0])
    items = sorted(filtered.items(), key=lambda x: (-x[1], x[0]))[:top_n]

    # STEP 7: Return top N keywords
    return items