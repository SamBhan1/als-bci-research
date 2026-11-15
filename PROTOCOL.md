# Evaluation protocol

This protocol defines the claims the repository is allowed to make.

## Primary question

How does P300 target detection generalize to people and recording sites not used during training, and how much labeled calibration is needed to improve transfer?

## Primary outcome

Participant-mean balanced accuracy on the independent NEMAR ALS test set. The mean is calculated after computing metrics separately for each participant. A pooled trial metric is secondary.

## Secondary outcomes

- Participant-mean ROC-AUC
- Participant-mean macro F1
- Bootstrap 95% confidence intervals over participants
- Calibration-performance curve
- Per-participant results
- Character or row/column accuracy only when stimulus-code metadata is present and verified

## Splitting rules

1. A participant must appear in exactly one evaluation partition.
2. Preprocessing learned from data—xDAWN filters, covariance geometry, classifier parameters, and class balancing—must be fit on training data only.
3. Calibration data may be used only when the protocol explicitly says so.
4. A calibration result must never be reported as zero-shot performance.
5. Hyperparameter selection must use a training-side validation split or a pre-specified literature value.

## Model comparisons

The minimum comparison is:

1. Transparent logistic baseline
2. xDAWN plus shrinkage LDA
3. xDAWN covariance plus Riemannian tangent-space logistic regression

An EEGNet comparison is a future extension after the dataset includes enough independent clinical participants to make neural-network tuning defensible.

## Calibration curve

Run 0%, 20%, 60%, and 80% labeled calibration as the primary curve. A 90% condition is exploratory only because it leaves too little independent evaluation data. Calibration boundaries must be aligned to participant recording order and must not mix future evaluation epochs into calibration examples. At the spelling level, average repeated flashes for each character/code before fitting or evaluating; do not average arbitrary same-class epochs across characters.

## Claim boundaries

These experiments do not establish diagnosis, treatment benefit, clinical safety, communication usability, or regulatory approval. A true patient-facing study requires clinical collaborators, ethics review, prospective data collection, and a pre-specified safety and usability protocol.

