"""Conservative quality gate for speech-recognition text used as evidence."""
import re
import unicodedata

_FILLERS = {
    "uh", "um", "umm", "uhh", "er", "erm", "hmm", "hmmm", "mmm", "mm",
    "uhuh", "mmhm", "huh",
}
_NOISE_TOKEN = re.compile(r"(?i)^(?:s{2,}|h{2,}m+|m{3,}|a{3,}|o{3,}|e{3,}|r{2,}m+)$")


def is_meaningful_transcript_text(value, language=""):
    """Reject ASR artifacts and non-lexical noise without requiring English."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text or "�" in text or "\\x00" in text:
        return False
    letters = [ch for ch in text if ch.isalpha() or ch.isdigit()]
    if not letters:
        return False

    # Whisper sometimes emits unrelated scripts in a single short segment
    # (for example Korean plus Thai under a Norwegian language tag).
    non_latin_scripts = set()
    for ch in letters:
        if "LATIN" in unicodedata.name(ch, ""):
            continue
        name = unicodedata.name(ch, "").split()
        if not name:
            continue
        if name[0] == "CJK":
            non_latin_scripts.add("CJK")
        elif name[0] in {"HIRAGANA", "KATAKANA"}:
            non_latin_scripts.add("Japanese")
        elif name[0] == "HANGUL":
            non_latin_scripts.add("Hangul")
        else:
            non_latin_scripts.add(name[0])
    if len(non_latin_scripts) > 1:
        return False
    if len(letters) == 1 and non_latin_scripts and next(iter(non_latin_scripts)) != "CJK":
        return False

    # A long run of a single grapheme is a common hallucinated transcript.
    longest = 1
    run = 1
    for left, right in zip(letters, letters[1:]):
        if left.casefold() == right.casefold():
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    if len(letters) >= 6 and longest / len(letters) >= 0.35:
        return False

    # Catch repeated-script hallucinations even when combining marks vary.
    normalized_letters = [ch.casefold() for ch in letters]
    if len(normalized_letters) >= 12 and len(set(normalized_letters)) <= 3:
        return False

    tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    if not tokens:
        return False
    folded = [token.casefold().strip("’'") for token in tokens if token]
    if not folded:
        return False
    if all(token in _FILLERS or _NOISE_TOKEN.fullmatch(token) for token in folded):
        return False
    if len(folded) == 1 and _NOISE_TOKEN.fullmatch(folded[0]):
        return False
    if len(folded) >= 6:
        counts = {}
        for token in folded:
            counts[token] = counts.get(token, 0) + 1
        if max(counts.values()) / len(folded) >= 0.8 and len(counts) <= 2:
            return False
    return True
