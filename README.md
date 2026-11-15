# ALS BCI generalization baseline

Research code for cross-participant P300 EEG decoding aimed at assistive communication research. The repository contains no patient EEG files; download them only after reading the dataset terms in [DATA_NOTICE.md](DATA_NOTICE.md).

Research question:

> Does an EEG classifier that looks strong on randomly mixed trials still work when tested on a person it never saw during training?

Read [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) for prior ALS/P300 work and [PROTOCOL.md](PROTOCOL.md) for the claim and splitting rules used here.

This is a deliberately small first experiment for ALS assistive-communication research. It can run on synthetic EEG-like epochs for pipeline checks, or on real EDF+ EEG recordings from the public PhysioNet bigP3BCI dataset. The real loader extracts P300 target/non-target epochs from EEG and preserves participant IDs.

The project is not a diagnostic or clinical tool. It is a reproducible benchmark scaffold. The public recordings are not a clinical validation set and must not be used to diagnose or treat anyone.

## Main finding: calibration-free spelling that works for people with ALS

A P300 speller normally needs a calibration session before it can be used: the person copies known letters
for several minutes so the decoder can learn their brain response. For someone with ALS that is tiring, and
it has to be repeated. UMM (Sosulski & Tangermann, 2023) is a published method that needs no calibration at
all. It scores above 99% on healthy users, but its authors report 77.86% on this ALS dataset, with some
participants at 0%, and explain why: in a row/column speller, one early wrong guess can poison everything it
remembers afterwards.

This repository tests a small fix: start UMM's memory from the average brain response of **other** ALS
patients instead of from nothing. The new user still provides no calibration and no labels.

Eight people with ALS, 35 letters each, 10 flash sequences per letter, exact-letter accuracy (chance is 2.8%):

| Method | Calibration needed from the user | All 35 letters | Worst participant | Letters 16-35 |
|---|---|---:|---:|---:|
| Supervised LDA (the usual way) | 15 letters | not usable for the first 15 | | 95.0% |
| Transfer only (other 7 patients, no adapting) | none | 76.1% | 57.1% | 73.8% |
| UMM, current letter only (reimplemented) | none | 38.9% | 14.3% | 40.6% |
| UMM with memory (reimplemented) | none | 59.6% | **0.0%** (two participants) | 66.2% |
| **Seeded UMM (this repository)** | **none** | **86.4%** (95% CI 82.5-90.0) | **77.1%** | **96.3%** |

- The failure the UMM authors described is reproduced (two participants at 0%), and seeding removes it: every
  participant is at 77% or better.
- After the first 15 letters, with no calibration, seeded UMM is as accurate as a decoder calibrated on those
  15 letters (96.3% vs 95.0%). Both targets set in advance in [PROTOCOL.md](PROTOCOL.md) were met.
- It beats either ingredient alone: other patients' data without adapting (76.1%) and adapting without other
  patients' data (59.6%).

Read this before quoting it:

- **Eight participants.** The paired tests are not statistically significant at this size (seeded vs UMM with
  memory p = 0.20; vs transfer only p = 0.06; vs supervised p = 0.63). The intervals above are over participants.
- **The UMM numbers are a reimplementation** from the paper's equations with this repository's preprocessing,
  not the authors' code. It gives 59.6% where they report 77.86%, so the gain over *their* number is about
  9 points, not 27.
- **It needs enough flashes.** With 5 or fewer sequences per letter, seeded UMM is no better than transfer
  only, and with 1-3 it is worse. See [RESULTS.md](RESULTS.md).
- **On the first 15 letters** seeded UMM gets 73.3% while it is still adapting.
- This is an offline replay of recorded sessions from one lab and one EEG setup. It is not a test with a
  person using the speller live, and it is not evidence of clinical benefit.

Reproduce it (about two minutes):

```cmd
.venv\Scripts\python.exe download_nemar_source.py
.venv\Scripts\python.exe calibration_free.py
.venv\Scripts\python.exe -m pytest -q
```

## Run

From the project folder in Command Prompt:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe experiment.py
```

Download a small Study L subset and run it:

```cmd
.venv\Scripts\python.exe download_dataset.py
.venv\Scripts\python.exe experiment.py --data-dir data\raw
```

Run the advanced comparison with ten same-class epochs averaged into each example:

```cmd
.venv\Scripts\python.exe advanced_experiment.py --data-dir data\raw --repeats 10
```

Save a paper-ready participant-disjoint benchmark as JSON:

```cmd
.venv\Scripts\python.exe benchmark.py --data-dir data\raw --repeats 10 --output benchmark_results.json
```

Download the independent ALS-only external-validation dataset and evaluate cross-dataset transfer:

```cmd
.venv\Scripts\python.exe download_nemar.py
.venv\Scripts\python.exe external_validation.py --train-dir data\raw --test-dir data\nm000169 --repeats 10
```

For the actual spelling-level metric, download the source-format stimulus codes and run:

```cmd
.venv\Scripts\python.exe download_nemar_source.py
.venv\Scripts\python.exe source_character_eval.py --source-dir data\nm000169\sourcedata --training-source calibration --calibration-fraction 0.6 --repeats 10 --model xdawn_lda
```

This evaluates exact row/column-pair selection per character using the source metadata. It is the appropriate communication-level metric; binary target detection should not be presented as spelling accuracy.

Use `--model logistic` to compare the unaligned baseline. The default `riemann` model uses xDAWN covariance alignment before tangent-space classification.

To measure a realistic short calibration phase, add for example `--calibration-fraction 0.2`. That uses the first 20% of each NEMAR participant's labeled epochs for adaptation and evaluates only on the remaining 80%; it is not zero-shot performance.

Observed results on the downloaded data:

- Study L participant-held-out: xDAWN covariance + Riemannian tangent space, 10-epoch averaging, balanced accuracy 0.703 and ROC-AUC 0.794.
- Independent NEMAR ALS external test: zero-shot balanced accuracy 0.689 and ROC-AUC 0.757.
- NEMAR ALS external test with 60% labeled calibration: participant-mean balanced accuracy 0.845 (95% bootstrap CI 0.811--0.878) and ROC-AUC 0.933 (95% CI 0.896--0.965).
- NEMAR ALS external test with 80% labeled calibration: participant-mean balanced accuracy 0.871 (95% bootstrap CI 0.845--0.903).
- NEMAR ALS external test with 90% labeled calibration: participant-mean balanced accuracy 0.882 (95% bootstrap CI 0.818--0.939), but only 10% of each participant remained for evaluation.
- Corrected source-format spelling evaluation (re-run 15 Nov 2026 after fixing a one-sample letter-boundary error in `source_character_eval.py`): xDAWN + shrinkage LDA reached 0.911 at 80% calibration (56 held-out characters) and 0.969 at 90% calibration (32 held-out characters, exploratory). Before the fix these were reported as 0.844 and 0.906.

The 80% calibration result is the strongest primary spelling result, but it still requires substantial labeled calibration. The 90% result uses a small held-out character set; it is exploratory and must not be presented as a general claim.

The NEMAR event table does not include the stimulus identity/order needed to reconstruct row/column characters, so `character_validation.py` is retained as an exploratory diagnostic only and is not a valid communication-accuracy result. A true character-level study requires a dataset that preserves stimulus codes.

Some PhysioNet files expose stimulus codes, but the encoding differs across Study L participants; the exploratory sequence decoder is therefore not used for a headline result until the codebook is verified per study.

This comparison uses participant-disjoint validation, training-only class balancing, the preregistered 250--500 ms P300 window, xDAWN spatial filtering, shrinkage LDA, bootstrap confidence intervals, and a paired Wilcoxon test. The ten-epoch result is closer to a BCI decision unit than a single stimulus response; it must not be reported as single-epoch accuracy.

The advanced script compares three appropriate models:

- time-resolved logistic regression as a transparent baseline;
- xDAWN spatial filtering plus shrinkage LDA;
- xDAWN-enhanced covariance features plus a Riemannian tangent-space classifier.

The last two are preferred for the current small, heterogeneous P300 dataset. A large transformer is not the default here because only 11 participants have been downloaded so far; it would make overfitting easier to hide.

The NEMAR external set contains eight people with ALS, eight EEG channels, and a 6x6 P300 copy-spelling task. It is licensed CC-BY-NC-ND, so keep it for analysis and do not redistribute it or derivative files.

The script prints accuracy, balanced accuracy, and macro F1 for:

- a random trial split, which can leak subject-specific patterns;
- leave-one-subject-out evaluation, which tests generalization to unseen people.

The current real-data pipeline applies a 0.5--30 Hz EEG filter, subtracts the pre-stimulus baseline, and extracts time-resolved features from ten post-stimulus windows. On the downloaded Study L subset, the current unseen-participant baseline is balanced accuracy 0.596 and macro F1 0.515. These are research baselines, not clinical performance claims.

## Research discipline

- Split by participant before fitting preprocessing or models.
- Report per-subject results, not only one average.
- Compare calibration time and performance on unseen participants.
- Treat this as a research result, not evidence for patient use.

