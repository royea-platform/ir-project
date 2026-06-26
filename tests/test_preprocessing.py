"""Tests for the preprocessing service."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.preprocessing.pipeline import preprocess_text, normalize_text


def test_normalize_basic():
    assert normalize_text("Hello WORLD!") == "hello world"


def test_normalize_urls():
    text = "Check https://example.com for info"
    result = normalize_text(text)
    assert "https" not in result
    assert "example" not in result


def test_preprocess_english():
    tokens = preprocess_text("What is machine learning?", language="en")
    assert len(tokens) > 0
    assert "what" not in tokens  # stopword removed
    assert "is" not in tokens    # stopword removed


def test_preprocess_stemming():
    tokens = preprocess_text("running quickly", language="en", use_stemming=True, remove_stopwords=False)
    assert "run" in tokens


def test_preprocess_lemmatization():
    tokens = preprocess_text("running dogs", language="en", use_stemming=False, use_lemmatization=True, remove_stopwords=False)
    assert "running" in tokens or "dog" in tokens


def test_deterministic():
    text = "Information retrieval systems"
    r1 = preprocess_text(text)
    r2 = preprocess_text(text)
    assert r1 == r2
