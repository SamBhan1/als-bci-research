# Calibration-free spelling (15 Nov 2026)

Produced by `calibration_free.py`; numbers are in `results/calibration_free_results.json`. Eight ALS
participants (BNCI 2014-008), exact-letter accuracy in %, participant mean. Protocol: PROTOCOL.md addendum.

All 35 letters, 10 sequences:

| Method | Mean | 95% CI over participants | Worst | Per participant (A01-A08) |
|---|---:|---|---:|---|
| transfer | 76.1 | 67.5-84.6 | 57.1 | 71, 60, 91, 74, 57, 83, 94, 77 |
| umm | 38.9 | 23.9-57.1 | 14.3 | 17, 40, 14, 26, 29, 31, 63, 91 |
| umm_memory | 59.6 | 31.1-85.0 | 0.0 | 83, 77, 31, 0, 0, 91, 94, 100 |
| **seeded_umm** | **86.4** | 82.5-90.0 | 77.1 | 77, 91, 89, 83, 83, 83, 94, 91 |

Letters 16-35 (after the supervised decoder's 15 calibration letters), 10 sequences:

| Method | Mean | 95% CI | Worst |
|---|---:|---|---:|
| supervised (15 labelled letters) | 95.0 | 91.9-97.5 | 85.0 |
| transfer | 73.8 | 61.3-85.6 | 45.0 |
| umm_memory | 66.2 | 35.6-92.5 | 0.0 |
| **seeded_umm** | **96.3** | 93.1-98.8 | 90.0 |

Letters 1-15 (a calibrated system is not usable yet): transfer 79.2, umm_memory 50.8, seeded_umm 73.3.

Paired Wilcoxon signed-rank, n = 8: seeded_umm vs umm_memory +26.8 points, p = 0.20; vs transfer +10.4,
p = 0.06; vs supervised on letters 16-35 +1.3, p = 0.63. With eight participants these tests have little power.

Accuracy by number of flash sequences, letters 16-35:

| Sequences | supervised | transfer | umm | umm_memory | seeded_umm |
|---:|---:|---:|---:|---:|---:|
| 1 | 14.4 | 17.5 | 4.4 | 3.8 | 11.2 |
| 2 | 35.0 | 40.0 | 8.1 | 6.2 | 20.0 |
| 3 | 60.6 | 46.2 | 11.2 | 13.1 | 42.5 |
| 5 | 77.5 | 59.4 | 17.5 | 36.9 | 76.9 |
| 7 | 88.1 | 72.5 | 30.6 | 62.5 | 89.4 |
| 10 | 95.0 | 73.8 | 40.6 | 66.2 | 96.2 |

Seeded UMM matches the calibrated decoder from 5 sequences up and is worse below that; with 1-2 sequences
plain transfer is the best calibration-free option.

Against the thresholds set in advance: "fixes the failure" (beats umm_memory, nobody under 50%) is met;
"removes the need for calibration" (within 5 points of supervised on letters 16-35) is met at 10 sequences.

Caveats: the UMM rows are a reimplementation and score below the 77.86% its authors report for this dataset;
eight participants from one site; offline replay, not live use.

**Correction.** `source_character_eval.py` assigned the first flash of each letter to the previous letter
(MATLAB 1-based index). Fixed on 15 Nov 2026. Re-running the xDAWN + shrinkage-LDA spelling evaluation gives
0.911 at 80% calibration (56 characters; participant CI 0.821-1.000) and 0.969 at 90% (32 characters;
0.906-1.000), replacing the 0.844 and 0.906 quoted further down. The other spelling numbers below (0.725,
0.734) were computed before the fix and have not been re-run.

# Current results

The strongest implemented model is xDAWN-enhanced covariance features mapped to a Riemannian tangent space and classified with balanced logistic regression. The evaluation uses 10 same-class epochs averaged into each training example and preserves participant-level splits.

## External ALS validation

Training data: PhysioNet Study L recordings.

External test data: eight ALS participants from NEMAR BNCI 2014-008.

| Protocol | Balanced accuracy | ROC-AUC |
|---|---:|---:|
| Zero-shot | 0.689 | 0.757 |
| 60% labeled calibration | 0.845 | 0.921 |
| 80% labeled calibration | 0.871 | 0.934 |
| 90% labeled calibration | 0.882 | 0.951 |

The 60% calibration result had participant-mean balanced accuracy 0.845 with a bootstrap 95% interval of 0.811--0.878. The 90% calibration result leaves only 10% of each participant's data for evaluation and is not a defensible 90% headline.

Using the official NEMAR source-format recordings, which preserve row/column stimulus codes, the corrected spelling-level protocol averaged repeated flashes within each character before decoding. Exact row/column-pair accuracy was 0.725 with 60% within-participant calibration (120 held-out characters; bootstrap 95% CI 0.642--0.800) and 0.734 with 80% calibration (64 held-out characters; CI 0.625--0.844).

After enforcing the 6x6 decoder structure (one row plus one column) and comparing the published xDAWN + shrinkage-LDA variant, exact row/column-pair accuracy reached 0.844 at 80% calibration (64 held-out characters; participant bootstrap CI 0.734--0.938) and 0.906 at 90% calibration (32 held-out characters; participant bootstrap CI 0.812--1.000). The 90% result is exploratory because the held-out character count is small.

## Interpretation

Calibration substantially improves transfer, but zero-shot generalization remains below the target. These numbers are research baselines, not clinical performance claims. Future work should add independent ALS participants, improve stimulus-code-aware models, and evaluate communication-level spelling accuracy under a preregistered protocol.

## References

- bigP3BCI dataset: https://doi.org/10.13026/0byy-ry86
- NEMAR BNCI 2014-008 ALS P300 dataset: https://doi.org/10.82901/nemar.nm000169
- xDAWN and Riemannian P300 transfer learning: https://doi.org/10.3390/app10051804
- EEGNet reference: https://doi.org/10.1088/1741-2552/aace8c

