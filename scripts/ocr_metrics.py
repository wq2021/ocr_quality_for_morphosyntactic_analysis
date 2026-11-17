"""Ground-truth-free text quality metrics used in both notebooks.

- OOV rate: proportion of words not found in the pyspellchecker French dictionary.
- Perplexity: perplexity of a small French GPT-2 model on the sentence.

Both functions are kept exactly as used for the results reported in the paper
(see "Known limitations" in README.md before changing them: any change will
alter the features and therefore the reported scores).
"""

import re

import numpy as np
import torch
from spellchecker import SpellChecker
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# OOV rate
# ---------------------------------------------------------------------------

spell = SpellChecker(language="fr")

# Short, frequent French words that are always considered valid,
# whatever the spellchecker dictionary says.
VALID_SHORT_WORDS = {
    "a", "y", "o", "à",
    "je", "tu", "il", "on",
    "le", "la", "un", "de", "du",
    "au", "en", "et", "ou", "où",
    "ce", "se", "me", "te", "ne",
    "si", "ni", "or",
    "qui", "que",
    "les", "des", "aux",
    "est", "ont",
    "son", "ton", "mon",
    "ces", "ses", "tes", "mes",
    "nos", "vos",
    "sur", "par", "pas", "peu",
    "car", "oui", "non",
    "ici", "là",
    "sa", "ta", "ma",
    "lui", "eux", "ils",
    "ans", "roi", "foi", "moi",
    "fut", "fit", "sut",
}

WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]+")


def compute_oov(text):
    """Return the OOV rate of `text` (rounded to 3 decimals), or None if not computable."""
    if not isinstance(text, str):
        return None

    words = WORD_PATTERN.findall(text.lower())
    if len(words) == 0:
        return None

    # NB: spell.unknown() returns a *set*, so a repeated unknown word is counted once.
    unknown = [w for w in spell.unknown(words) if w not in VALID_SHORT_WORDS]
    return round(len(unknown) / len(words), 3)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

LM_ID = "dbddv01/gpt2-french-small"

_tokenizer = None
_lm = None


def _load_lm():
    """Load the language model once, on first use."""
    global _tokenizer, _lm
    if _lm is None:
        _tokenizer = AutoTokenizer.from_pretrained(LM_ID)
        _lm = AutoModelForCausalLM.from_pretrained(LM_ID)
        _lm.eval()
    return _tokenizer, _lm


def compute_perplexity(text):
    """Return the perplexity of `text` (rounded to 1 decimal), or None if not computable."""
    if not isinstance(text, str) or text.strip() == "":
        return None

    tokenizer, lm = _load_lm()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = lm(**inputs, labels=inputs["input_ids"])
    return round(torch.exp(outputs.loss).item(), 1)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

# Order matters for the random forest (random_state is tied to column order).
FEATURE_SETS = {
    "OOV + PPL": ["Score_OOV_apres", "Perplexite_log"],
    "OOV Gain + PPL Gain": ["Gain_Score_OOV", "Gain_Score_Perplexite_log"],
    "All four features": [
        "Gain_Score_OOV",
        "Gain_Score_Perplexite_log",
        "Score_OOV_apres",
        "Perplexite_log",
    ],
}


def add_features(df, original_col="PivotSent", corrected_col="pivot_sent_corrected"):
    """Compute OOV rate and perplexity on the original and post-corrected pivot
    sentences, then the log-perplexities and the gains (original - corrected).

    Column names (French) are kept as in the annotated data files:
      *_avant = original OCR sentence, *_apres = OCRonos post-corrected sentence.
    """
    df = df.copy()

    df["Score_OOV_avant"] = df[original_col].apply(compute_oov)
    df["Score_Perplexite_avant"] = df[original_col].apply(compute_perplexity)
    df["Score_OOV_apres"] = df[corrected_col].apply(compute_oov)
    df["Score_Perplexite_apres"] = df[corrected_col].apply(compute_perplexity)

    df["Score_Perplexite_avant_log"] = np.log(df["Score_Perplexite_avant"])
    df["Perplexite_log"] = np.log(df["Score_Perplexite_apres"])

    df["Gain_Score_OOV"] = df["Score_OOV_avant"] - df["Score_OOV_apres"]
    df["Gain_Score_Perplexite_log"] = df["Score_Perplexite_avant_log"] - df["Perplexite_log"]
    return df
