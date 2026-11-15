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

