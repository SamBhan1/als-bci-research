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
- McCane et al., “An exploration of BCI performance variations in people with amyotrophic lateral sclerosis using longitudinal EEG data,” https://pmc.ncbi.nlm.nih.gov/articles/PMC7951329/
- Li et al., “Transfer Learning Algorithm of P300-EEG Signal Based on XDAWN Spatial Filter and Riemannian Geometry Classifier,” https://doi.org/10.3390/app10051804
- Lawhern et al., “EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces,” https://doi.org/10.1088/1741-2552/aace8c
- Mainsah et al., “bigP3BCI: An Open, Diverse and Machine Learning Ready P300-based Brain-Computer Interface Dataset,” https://doi.org/10.13026/0byy-ry86

