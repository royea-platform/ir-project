"""Spelling correction using spellchecker."""

from spellchecker import SpellChecker

_spell = None


def _get_spell() -> SpellChecker:
    global _spell
    if _spell is None:
        _spell = SpellChecker()
    return _spell


def correct_spelling(text: str) -> str:
    """Correct misspelled words in the query."""
    spell = _get_spell()
    words = text.split()
    corrected = []
    for word in words:
        correction = spell.correction(word)
        corrected.append(correction if correction else word)
    return " ".join(corrected)
