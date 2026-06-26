"""Synonym expansion using WordNet."""

from nltk.corpus import wordnet


def expand_synonyms(text: str, max_synonyms_per_word: int = 2) -> str:
    """Add top WordNet synonyms to query terms."""
    words = text.split()
    expanded = list(words)

    for word in words:
        try:
            synsets = wordnet.synsets(word)
        except Exception:
            continue
        added = 0
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()
                if synonym != word.lower() and synonym not in expanded:
                    expanded.append(synonym)
                    added += 1
                    if added >= max_synonyms_per_word:
                        break
            if added >= max_synonyms_per_word:
                break

    return " ".join(expanded)
