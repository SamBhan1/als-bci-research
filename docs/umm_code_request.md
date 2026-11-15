# Draft email: request for the UMM implementation

Not sent. The authors' addresses are shown as images in the README of
https://github.com/jsosulski/umm_demo (Jan Sosulski, Michael Tangermann). Their README asks for the subject
line below and for your name, affiliation and intended use. Fill in the bracketed parts before sending.

---

**Subject:** umm.py request

Dear Dr. Sosulski, dear Prof. Tangermann,

My name is [your name], [student / independent researcher] at [affiliation]. I would like to request
`umm.py` for **non-commercial, academic use only**.

I am studying calibration-free P300 spelling for people with ALS. Your paper reports 77.86% on the Riccio
et al. (2013) ALS dataset and explains the row/column failure mode when means are learned from earlier
letters. I reimplemented UMM from the paper's equations to test whether initialising the mean memory from
other participants removes that failure. The code and results are public:
https://github.com/SamBhan1/als-bci-research

My reimplementation does not reproduce your figure. With the preprocessing from your demo notebook
(0.5-8 Hz, 40 Hz, 0.05-0.7 s, Ledoit-Wolf shrinkage) the best of the variants described in the paper reaches 69.6% on
that dataset (pooled covariance, confidence-weighted means). I would like to replace my baseline with your implementation so the
comparison is fair to your method, and I will cite the paper and state clearly that the code was provided
by you.

Three questions, if you have a moment:

1. Which configuration (covariance data, covariance estimator, mean estimation) gave the 77.86%?
2. Was Ric13 preprocessed like the Huebner datasets in the demo, or differently (channels, interval,
   epochs rejected)?
3. Is the result an average over participants, and over all 35 letters each?

I understand the code is under Radboud University copyright with a patent pending, and I will not
redistribute it or use it outside this research.

Thank you for making the paper and the demo available.

Kind regards,
[your name]
[email]
