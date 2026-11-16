# Prior work and the research gap

## What is already established

### ALS communication with P300 BCIs

P300 spellers for ALS are not a new idea. Nijboer et al. evaluated a P300 communication device in people with advanced ALS using a matrix speller. The work established the clinical motivation and showed that performance varies substantially between people.

Sellers et al. later evaluated 25 people with ALS using an 8-channel visual P300 BCI. A high-performing group averaged about 92% BCI accuracy, while another group performed poorly. Their result is important because it shows that a high headline number can coexist with serious user-to-user heterogeneity.

Longitudinal work has also studied nine people with ALS using P300 BCIs at home over several months. That work found day-to-day EEG features associated with successful performance, reinforcing that calibration and stability matter outside a laboratory session.

### Algorithms

The classical P300 pipeline is typically built around a spatial or temporal ERP enhancer followed by a regularized classifier. xDAWN is designed to enhance evoked responses. Riemannian covariance methods model EEG trial geometry and have been proposed for reducing cross-subject calibration burden. Shrinkage LDA and logistic regression remain valuable baselines because they are interpretable and work well in small-sample settings.

EEGNet is a compact convolutional model designed for EEG BCI paradigms, including P300. It is a useful future comparison, but a deep model should not be the default headline model with only a small number of clinical participants and heterogeneous recording protocols.

## What this project adds

The project should be framed as a reproducible benchmark and external-validation study, not as a new P300 algorithm. Its proposed contribution is:

> A leakage-controlled comparison of classical and Riemannian P300 decoders under participant-disjoint and cross-dataset ALS evaluation, reporting the tradeoff between zero-shot performance and labeled calibration.

The important gaps are:

1. Many BCI papers report within-dataset results; fewer stress-test transfer across recording studies and hardware.
2. Raw accuracy is not enough when target/non-target classes are imbalanced; balanced accuracy, ROC-AUC, macro F1, and participant-level intervals are required.
3. Calibration is often reported as a practical detail rather than a primary outcome. Here it is an explicit independent variable.
4. Communication-level character accuracy requires stimulus identity metadata. If a derivative only retains target/non-target labels, it must not claim spelling accuracy.

## Primary references

- Nijboer et al., “A P300-based brain-computer interface for people with amyotrophic lateral sclerosis,” https://doi.org/10.1016/j.clinph.2008.03.034
- Sellers et al., “Brain-computer interface (BCI) evaluation in people with amyotrophic lateral sclerosis,” https://pubmed.ncbi.nlm.nih.gov/24555843/
- Shahriari et al., “An exploration of BCI performance variations in people with amyotrophic lateral sclerosis using longitudinal EEG data,” https://pmc.ncbi.nlm.nih.gov/articles/PMC7951329/
- Li et al., “Transfer Learning Algorithm of P300-EEG Signal Based on XDAWN Spatial Filter and Riemannian Geometry Classifier,” https://doi.org/10.3390/app10051804
- Lawhern et al., “EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces,” https://doi.org/10.1088/1741-2552/aace8c
- Mainsah et al., “bigP3BCI: An Open, Diverse and Machine Learning Ready P300-based Brain-Computer Interface Dataset,” https://doi.org/10.13026/0byy-ry86

## Calibration-free spelling (added 15 Nov 2026)

- **Kindermans et al. (2014)** combined transfer learning between users, unsupervised adaptation, a language
  model and dynamic stopping into a zero-training speller, tested in simulation on 22 healthy-user recordings.
  https://doi.org/10.1088/1741-2560/11/3/035005
- **Mainsah et al. (2015)** showed online, with ALS participants, that dynamic stopping raises communication
  rate while keeping accuracy; the decoder was still calibrated. https://pmc.ncbi.nlm.nih.gov/articles/PMC4631027
- **UMM (Sosulski & Tangermann, 2023)** needs no calibration and no other users' data: for each candidate
  symbol it computes the target-minus-non-target mean difference that symbol would imply and picks the largest
  covariance-corrected distance. It reports above 99% on healthy-user visual spellers and 77.86% on the ALS
  dataset used here (Riccio et al., 2013), with some participants at 0% when it learns from earlier letters.
  The authors attribute this to the row/column layout: after an early wrong decision the remembered means can
  point the wrong way, and a distance cannot see direction. https://arxiv.org/abs/2306.11830
  (demo code: https://github.com/jsosulski/umm_demo)
- **Riccio et al. (2013)** recorded the dataset: eight people with ALS, 8 channels, 6x6 row/column speller,
  35 letters each. https://doi.org/10.3389/fnhum.2013.00732

**The gap.** The method that removes calibration fails for exactly the users who most need calibration
removed. No public repository was found (web and GitHub search, 14-15 Nov 2026) that evaluates
calibration-free spelling on ALS data against a calibrated baseline letter by letter, or that tests whether
other patients' data repairs UMM's row/column failure. `calibration_free.py` does both. The idea of seeding
from other users is Kindermans et al.'s; what is new here is applying it to UMM's memory and measuring it on
people with ALS.

## After the extended study (added 15 Nov 2026)

- The UMM authors' public repository contains the preprocessing and the replay loop but not the algorithm
  file, which is available on request. The extended study therefore copies their preprocessing settings
  (0.5-8 Hz, 40 Hz, 0.05-0.7 s) and keeps a reimplemented decision rule.
- **Arico et al. (2014)** recorded BNCI 2014-009: ten healthy users on the same row/column speller in the
  same lab as the ALS dataset. https://doi.org/10.1088/1741-2560/11/3/035008
- **Mainsah et al. (2025), bigP3BCI**: Study L holds eleven people with ALS on a checkerboard speller recorded
  at a different site. Checkerboard flashing (Townsend et al., 2010) lights scattered groups of cells instead
  of whole rows and columns. https://doi.org/10.13026/0byy-ry86
- What the extended study adds to the gap above: the seeding effect holds across two ALS cohorts, and the
  seed fails when it comes from a different recording setup. Cross-setup transfer for ERP spellers is usually
  handled by aligning recordings first (for example Riemannian re-centring; see the xDAWN/Riemannian
  reference above); whether that repairs the seed was not tested here.

## After the timing study (added 15 Nov 2026)

- P300 latency is known to be longer and more variable in people with ALS than in healthy users, and latency
  jitter lowers speller accuracy (Arico et al., 2014, above). The three datasets here do peak at different
  times. Shifting a borrowed seed to line up the peaks nevertheless made it worse in every case, so the
  cross-setup failure is not explained by delay alone (LATENCY_STUDY.md).
