from __future__ import annotations

import re
from typing import Iterable, List

STOPWORDS = {
    "a","an","and","are","as","at","be","been","being","but","by","for","from",
    "had","has","have","he","her","hers","him","his","i","if","in","into","is",
    "it","its","itself","me","my","of","on","or","our","ours","she","so","that",
    "the","their","them","they","this","those","to","too","up","was","we","were",
    "what","when","which","who","will","with","you","your","yours","very","just",
    "than","then","there","here","about","after","before","because","while","during"
}

TEXT_COLUMN_CANDIDATES = [
    "text","review","content","comment","feedback","tweet","message","post","caption","body","sentence"
]

POSITIVE_HINTS = {
    "love","loved","great","amazing","excellent","fantastic","good","best","wonderful",
    "enjoyed","impressed","smooth","delicious","satisfying","happy","helpful","recommend",
    "premium","intuitive","quick","useful","pleasant","solid","friendly"
}
NEGATIVE_HINTS = {
    "hate","hated","awful","terrible","bad","worst","boring","messy","slow","broken",
    "bug","bugs","disappointed","cold","late","damaged","weak","poor","confusing",
    "annoying","frustrating","difficult","horrible","useless","cheap","rude","long"
}
NEUTRAL_HINTS = {
    "okay","fine","average","decent","standard","acceptable","basic","ordinary",
    "mixed","some","manageable","reasonable","serviceable","typical"
}
NEGATION_MAP = {
    "not bad": "good",
    "not good": "bad",
    "not great": "average",
    "not terrible": "okay",
    "not awful": "okay",
    "not amazing": "average",
    "not happy": "sad",
    "not satisfied": "disappointed",
}

def normalize_text(text: str) -> str:
    # Kung None ang input, ibalik empty string para malikayan errors
    if text is None:
        return ""
    
    # Convert to string, tangtang extra spaces, ug himuon lowercase tanan
    text = str(text).strip().lower()
    
    # Replace negation phrases (e.g. "not good" → "not_good")
    # para mas klaro ang sentiment meaning
    for phrase, repl in NEGATION_MAP.items():
        text = text.replace(phrase, repl)
    
    # Remove URLs (http, www links)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # Remove special characters (retain letters, numbers, space, apostrophe)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    
    # Normalize spaces (multiple spaces → single space)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def tokenize(text: str) -> List[str]:
    # Step 1: I-normalize ang text (cleaning)
    text = normalize_text(text)
    
    # Step 2: Split into tokens (words)
    # Step 3: Remove empty tokens ug stopwords (common words like "the", "is")
    tokens = [tok for tok in text.split() if tok and tok not in STOPWORDS]
    
    return tokens


def preprocess_text(text: str) -> str:
    # Convert tokens balik into cleaned string
    # (useful for TF-IDF or ML models)
    #join balik ang words na gi tokenize
    return " ".join(tokenize(text))


def detect_text_column(df):
    # Kuhaon tanan column names
    cols = list(df.columns)
    
    # Create mapping: lowercase column name → original column name
    # para dali ma-match bisan lahi ang casing
    lowered = {str(c).strip().lower(): c for c in cols}
    
    # Step 1: Try ug detect predefined text column names
    # (e.g. "text", "message", "review", etc.)
    for cand in TEXT_COLUMN_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    
    # Step 2: Kung walay match, pili og column nga string/object type
    object_cols = [c for c in cols if str(df[c].dtype) in ("object", "string")]
    
    if object_cols:
        # Pili ang column nga adunay pinakataas average length
        # (assumption: mao ni ang main text column)
        best = max(object_cols, key=lambda c: df[c].astype(str).str.len().mean())
        return best
    
    # Step 3: Fallback → first column kung walay lain choice
    return cols[0] if cols else None