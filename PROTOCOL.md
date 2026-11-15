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


## Addendum (15 Nov 2026): calibration-free spelling

Written before `calibration_free.py` was run on any data.

**Question.** UMM (Sosulski & Tangermann 2023) decodes ERP spellers with no calibration and reports over 99%
on healthy users but 77.86% on this ALS dataset, with some participants at 0% when it learns from earlier
letters. The authors' explanation: in a row-column speller, one early wrong decision can leave the remembered
class means pointing the wrong way, and every later letter inherits it. Does starting that memory from other
ALS patients' average responses remove the failure, with no calibration from the new user?

**Data.** The eight ALS participants of BNCI 2014-008, source files, 35 letters each, 10 sequences of 12
flashes per letter.

**Fixed in advance (not tuned):** 0.5-16 Hz band-pass; 0-800 ms after each flash; baseline -100-0 ms; mean
of 50 ms bins on 8 channels (128 features); Ledoit-Wolf shrinkage covariance; seed weight equal to five
confident letters; supervised reference trained on the participant's first 15 letters.

**Methods.** `supervised`, `transfer` (other seven patients, no adaptation), `umm`, `umm_memory`,
`seeded_umm`, as described at the top of `calibration_free.py`. For a held-out participant, everything
learned from other people uses the other seven only. No label of the held-out participant is used by any
calibration-free method.

**Primary outcome.** Participant-mean exact-letter accuracy (chance 1/36) over all 35 letters with 10
sequences; bootstrap 95% interval over participants; lowest participant; paired Wilcoxon signed-rank tests
(n = 8, so low power: a non-significant p is not evidence of no difference).

**Secondary.** The same with 1, 2, 3, 5 and 7 sequences; accuracy on the last 20 letters next to the
supervised reference; accuracy on the first 15 letters, when a calibrated system is still unusable.

**What would count as a result.** The seeding "fixes the failure" if `seeded_umm` beats `umm_memory` on the
primary outcome and no participant is under 50%. It "removes the need for calibration" only if, on the last
20 letters, its participant mean is within 5 points of `supervised`. Whatever happens is reported, including
if plain `transfer` does as well as `seeded_umm`, which would mean the UMM part adds nothing.

**Limits known in advance.** Eight participants. Offline simulation of online use. The reimplementation of
UMM follows the paper's equations but is not the authors' code or preprocessing, so it need not reproduce
77.86% exactly. The seed comes from the same recording setup and site; transfer across hardware is untested.

## Addendum 2 (15 Nov 2026): replication, authors' preprocessing, seeds from elsewhere

Written after the first calibration-free run (addendum above) and before any of the analyses below were run.
The first run had three weaknesses: eight participants, a UMM reimplementation that scored below the
published figure, and a seed taken from the same lab and setup. This addendum fixes what will be tested.

**Datasets** (all 256 Hz, restricted to the same eight channels Fz, Cz, Pz, Oz, P3, P4, PO7, PO8; 6x6 matrix):

| Name | Source | People | Paradigm | Letters each | Flashes per letter |
|---|---|---|---|---|---|
| `rome_als` | BNCI 2014-008 | 8 with ALS | row/column | 35 | 120 |
| `rome_healthy` | BNCI 2014-009 | 10 healthy | row/column | 18 | 96 |
| `etsu_als` | bigP3BCI Study L, checkerboard training files | 11 with ALS | checkerboard | 30 (one person 24) | 126 |

`etsu_als` was recorded at a different site with different hardware and a different flashing pattern. Which
cells flashed is read from the per-cell indicator channels; the target is `CurrentTarget`.

**Methods** as before, generalised from row/column codes to "which cells lit up", plus
`seeded_umm[<other dataset>]`: the seed comes from every participant of a *different* dataset, so nothing
from the test participant's lab, hardware or (for `rome_healthy`) patient group is used.
The supervised reference trains on the first 15 (`rome_als`), 6 (`rome_healthy`) or 12 (`etsu_als`) letters.

**Preprocessing.** `binned` is the pipeline fixed in the first addendum and stays primary. `authors` copies
the settings in the UMM authors' public demo: 0.5-8 Hz, resampled to 40 Hz, 0.05-0.7 s, every sample as a
feature, no baseline correction. The authors' own algorithm file is not public (available on request only),
so UMM remains a reimplementation of the paper's equations under both pipelines.

**Questions and what counts as an answer**

1. *Is the weak UMM baseline an artefact of my preprocessing?* Report `umm_memory` on `rome_als` under
   `authors`. If it lands within 5 points of the published 77.86%, the reimplementation is taken as faithful.
   Whatever it scores, seeded UMM is compared against it under the same pipeline.
2. *Does the effect replicate in other people with ALS?* On `etsu_als`, `seeded_umm` (seed from the other
   ten) must beat `umm_memory` on participant-mean accuracy.
3. *Does seeding do harm where UMM already works?* On `rome_healthy`, `seeded_umm` must be no more than
   3 points below `umm_memory`.
4. *Can the seed come from elsewhere?* For `rome_als`: seed from `rome_healthy` (other patient group) and from
   `etsu_als` (other site and paradigm). For `etsu_als`: seed from `rome_als`. Each counts as working if it
   beats `umm_memory` and is within 5 points of the same-dataset seed.
5. *Primary confirmatory test.* Across all 19 people with ALS (both sites), `seeded_umm` vs `umm_memory`,
   all letters, all flashes, `binned` pipeline: exact two-sided sign-flip permutation test on participant
   differences, alpha 0.05, with a participant-bootstrap 95% interval for the mean difference. This is the
   one test the headline claim rests on. Everything else is reported with intervals and is secondary.

**Statistics.** Participant is the unit. Intervals: 10,000-resample bootstrap over participants. Paired
comparisons: exact sign-flip permutation (all 2^n assignments), plus Wilcoxon for continuity with the first
addendum. No correction for multiple comparisons is applied to secondary results; they are labelled as such.

**Known limits.** Still offline replays. `etsu_als` checkerboard training files only. The eight shared
channels discard half of the electrodes available in two of the datasets. Seeds and seed weight are not tuned.

**Deviation recorded before any addendum-2 result was seen (15 Nov 2026).** While loading the data, the
healthy Rome files turned out to be stored at roughly seven times the amplitude of the ALS files, so a seed
taken from one dataset would be in the wrong units for another. The seed is therefore stored unit-free (each
contributing person divided by their own overall standard deviation) and rescaled by the standard deviation of
the new user's own epochs seen so far. This needs no labels and no future data. It is applied to every seeded
variant, including the same-dataset seed, so the `rome_als` figure can differ slightly from the first run
(86.4%), which used raw units. A first launch of the addendum-2 run was stopped before finishing and its
output was not looked at.
