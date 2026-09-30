"""Whole-word query hints, preferring longer phrases over overlapping words."""
import re
import unicodedata


def keyword_hints(text, vocabulary):
    normalized = unicodedata.normalize('NFC', text or '').casefold()
    words = {unicodedata.normalize('NFC', word).casefold(): value
             for word, value in vocabulary.items() if word}
    if not words:
        return set()
    alternatives = '|'.join(re.escape(word) for word in sorted(words, key=lambda w: (-len(w), w)))
    return {words[match.group()] for match in re.finditer(r'(?<!\w)(?:' + alternatives + r')(?!\w)', normalized)}
