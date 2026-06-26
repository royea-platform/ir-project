"""Text preprocessing pipeline — shared between document indexing and query processing."""

import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Ensure NLTK data available (skip download if already present)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True, raise_on_error=True)
        nltk.download("stopwords", quiet=True, raise_on_error=True)
        nltk.download("wordnet", quiet=True, raise_on_error=True)
    except Exception:
        pass

# Cache stemmers and stopwords
_stemmers = {}
_stopwords = {}
_lemmatizer = None


def _get_stemmer(language: str):
    if language not in _stemmers:
        lang_map = {"en": "english", "ar": "arabic"}
        _stemmers[language] = SnowballStemmer(lang_map.get(language, "english"))
    return _stemmers[language]


def _get_stopwords(language: str) -> set:
    if language not in _stopwords:
        lang_map = {"en": "english", "ar": "arabic"}
        _stopwords[language] = set(stopwords.words(lang_map.get(language, "english")))
    return _stopwords[language]


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def normalize_text(text: str) -> str:
    """Unicode NFKC normalization + lowercasing + cleaning."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(
    text: str,
    language: str = "en",
    use_stemming: bool = True,
    use_lemmatization: bool = False,
    remove_stopwords: bool = True,
) -> list[str]:
    """Full preprocessing pipeline: normalize → tokenize → stopwords → stem/lemmatize."""
    # Normalize
    text = normalize_text(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    if remove_stopwords:
        stop = _get_stopwords(language)
        tokens = [t for t in tokens if t not in stop]

    # Remove short tokens
    tokens = [t for t in tokens if len(t) > 1]

    # Stemming or lemmatization (not both)
    if use_lemmatization:
        lemmatizer = _get_lemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    elif use_stemming:
        stemmer = _get_stemmer(language)
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens
