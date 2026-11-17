# Assessing OCR Quality for Morphosyntactic Analysis in a French Corpus

Code for the paper:

> Qi Wang, Mélanie Barret, Astrid Chevance, Frédérique Mélanie-Becquet.
> *Assessing OCR Quality for Morphosyntactic Analysis in a French Corpus: Methodology and Application.*
> Computational Humanities Research (CHR 2027), Anthology of Computers and the Humanities. <!-- TODO: volume, pages, DOI -->

The work is part of the [GénéAlgoPsy](https://zenodo.org/records/7845152) project, which built a corpus of about
three million concordance examples around the French word *douleur* (pain) from Gallica, the digital library of the
Bibliothèque nationale de France (BnF).

## Overview

Gallica texts are produced by OCR, and transcription noise degrades the morphosyntactic analysis of the corpus
(POS tagging, lemmatization, dependency parsing). Instead of measuring transcription accuracy, we ask whether a pivot
sentence is **reliable enough for morphosyntactic analysis**, without any gold-standard transcription.

The pipeline:

1. Each pivot sentence (the sentence containing *douleur*, *douloureux* or *douloureusement*) is post-corrected with
   [OCRonos](https://huggingface.co/PleIAs/OCRonos).
2. Two annotators label 1,134 original/corrected sentence pairs: class 1 = analyzable, class 0 = not analyzable.
3. Two ground-truth-free metrics are computed on the original and corrected sentences:
   - **OOV rate** with [`pyspellchecker`](https://pypi.org/project/pyspellchecker/) (French dictionary);
   - **perplexity** with a small French GPT-2 model ([`dbddv01/gpt2-french-small`](https://huggingface.co/dbddv01/gpt2-french-small)).
4. Logistic regression and random forest classifiers are trained on three feature sets; the final model is
   logistic regression with four features (OOV rate and log-perplexity of the corrected sentence, and their gains,
   original − corrected).
5. The final model is applied to all pivot sentences of the 34 documents that contain 800–899 examples, and the
   predicted noise rate of each document is compared with the BnF document-level OCR score.

## Repository structure

```
.
├── data/                                # not included
│   ├── annotations_withPeriod.csv       # 1,134 annotated examples
│   └── slices_800-900.tsv               # 28,488 examples from the 34 documents with 800–899 examples
├── scripts/
│   ├── ocr_metrics.py                   # OOV rate, perplexity and feature construction
│   ├── 1_train_classifier.ipynb         # Tables 1–4, trains and saves the final model
│   └── 2_document_level_analysis.ipynb  # Table 5, Figures 1–2, correlations
└── outputs/                             # created when the notebooks are run
```

## Data

The annotated data are derived from Gallica documents and are not distributed in this repository.

The notebooks expect two tab-separated files in `data/`:
`annotations_withPeriod.csv` and `slices_800-900.tsv`, with one row per example and the columns
`doc_id`, `PivotSent`, `pivot_sent_corrected`, `Date_correction`, plus
`annot_preCorrection`, `annot_postCorrection`, `annot`, `period` (annotations) or `OCR` (slices).


## Installation

Python ≥ 3.10.

```bash
pip install pandas numpy scipy matplotlib joblib torch transformers pyspellchecker scikit-learn==1.9.0
pip install jupyter
```

The results in the paper were obtained with scikit-learn 1.9.0. The GPT-2 model is downloaded from the Hugging Face Hub on first use.

## Usage

Run the notebooks from the `scripts/` directory, in order:

1. **`1_train_classifier.ipynb`**: computes the features for the annotated data, compares the six classifier
   configurations, evaluates the final model and saves it to `outputs/best_model.pkl`.
2. **`2_document_level_analysis.ipynb`**: loads the saved model, predicts the class of every pivot sentence of the
   34 documents, and computes the per-document noise rates and their correlation with the BnF OCR score.

Computing perplexity is the slowest step (about 57,000 sentences in the second notebook); a GPU helps but is not
required.

## Limitations

The metric functions are kept exactly as used for the published results; changing them will change the scores.

- The OOV rate counts each distinct unknown word once (`spell.unknown()` returns a set), while the denominator
  counts all word tokens.
- The word pattern `[A-Za-zÀ-ÿ]` does not include `œ`/`Œ`, so words such as *cœur* are split into fragments
  that are counted as unknown.
- The perplexity model was trained on contemporary French Wikipedia and may not represent earlier periods well.
- Per-period noise rates are means over documents, not over pooled examples.




## Acknowledgements

This work received support from the UROP@PSL internship program and from the GénéAlgoPsy project, which benefits
from methodological and financial support from the BnF through the Huma-Num DataLab call for proposals.
We thank the BnF DataLab team for their help in designing and building the corpus.
