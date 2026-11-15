# ALS BCI generalization baseline

Research code for cross-participant P300 EEG decoding aimed at assistive communication research. The repository contains no patient EEG files; download them only after reading the dataset terms in [DATA_NOTICE.md](DATA_NOTICE.md).

Research question:

> Does an EEG classifier that looks strong on randomly mixed trials still work when tested on a person it never saw during training?

Read [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) for prior ALS/P300 work and [PROTOCOL.md](PROTOCOL.md) for the claim and splitting rules used here.

This is a deliberately small first experiment for ALS assistive-communication research. It can run on synthetic EEG-like epochs for pipeline checks, or on real EDF+ EEG recordings from the public PhysioNet bigP3BCI dataset. The real loader extracts P300 target/non-target epochs from EEG and preserves participant IDs.

The project is not a diagnostic or clinical tool. It is a reproducible benchmark scaffold. The public recordings are not a clinical validation set and must not be used to diagnose or treat anyone.

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
- Corrected source-format spelling evaluation: xDAWN + shrinkage LDA reached 0.844 at 80% calibration and 0.906 at 90% calibration; the latter used only 32 held-out characters and is exploratory.

The 80% calibration result is the strongest primary spelling result, but it still requires substantial labeled calibration. The 90% result crosses 90% on the small held-out character set, but is exploratory and must not be presented as a general 90% claim.

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

