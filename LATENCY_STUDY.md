# Why the seed fails across setups: a timing correction that did not work

Produced by `latency_study.py`; numbers are in `results/latency_study_results.json`. The plan was written
first: [PROTOCOL.md](PROTOCOL.md), addendum 3. **Both things tried here failed.** They are reported because a
failed, pre-specified test is still a result, and because it rules out the obvious explanation.

## What the responses look like (descriptive)

Average target-minus-non-target response per dataset, `binned` features, each person scaled to unit variance.

| Dataset | Peak at Pz after the flash marker |
|---|---:|
| Rome healthy | about 375 ms |
| Rome ALS | about 525 ms |
| ETSU ALS | about 225 ms |

| Pair | Similarity at zero shift | Best shift of the second | Similarity at best shift |
|---|---:|---:|---:|
| Rome ALS vs Rome healthy | 0.46 | +100 ms | 0.83 |
| Rome ALS vs ETSU ALS | -0.28 | +250 ms | 0.55 |
| Rome healthy vs ETSU ALS | 0.05 | +150 ms | 0.62 |

So the three setups disagree about *when* the response happens, and at zero shift the two ALS sites are
negatively related. That made timing the obvious suspect.

## Test 1 (pre-specified): latency-searched seed

The borrowed seed is tried at 13 shifts from -300 to +300 ms; the shift with the highest running UMM
confidence makes each decision. No labels, no future data.

New-clinic scenario: each of the 19 people with ALS seeded only from the **other** site's ALS participants.

| Method | Accuracy (95% CI) |
|---|---:|
| UMM with memory, no seed | 61.1% (43.8-77.9) |
| Borrowed seed, unshifted | 44.7% (28.0-61.0) |
| **Borrowed seed, latency search** | **40.6%** (22.7-59.9) |
| Local seed, for reference | 78.4% (67.6-87.1) |

- Against no seed: **-20.5 points** (95% CI -36.0 to -6.2; exact sign-flip p = 0.014). The pre-specified test
  came out significantly in the wrong direction. A new clinic is better off with no seed than with another
  lab's seed, searched or not.
- Against the unshifted borrowed seed: -4.1 points (CI -25.6 to +17.7). No improvement.
- It also hurt when the seed was local, where no shift was needed: -10.7 points on Rome ALS, -7.8 on Rome
  healthy, -32.7 on ETSU ALS. The limit set in advance was 3 points.
- The shifts it settled on were scattered (for example -250 to +250 ms across ETSU participants given the
  same borrowed seed). UMM's confidence is not a usable signal for choosing a shift.

## Test 2 (post-hoc): is timing even the cause?

To test the explanation rather than the method, the borrowed seed was shifted by one site-level number,
chosen with the labels of *other* people at the test participant's site. This is not a usable
calibration-free method; it is the most favourable version of "just fix the timing".

| Test group | Borrowed from | Unshifted | Shifted by the site-level offset | Local seed |
|---|---|---:|---:|---:|
| Rome ALS | Rome healthy | 73.9% | 60.0% | 86.8% |
| Rome ALS | ETSU ALS | 64.6% | 15.7% | 86.8% |
| ETSU ALS | Rome ALS | 30.1% | 22.4% | 72.4% |
| ETSU ALS | Rome healthy | 40.1% | 28.0% | 72.4% |
| Rome healthy | Rome ALS | 92.2% | 78.9% | 94.4% |
| Rome healthy | ETSU ALS | 86.7% | 61.7% | 94.4% |

Shifting made every case worse. **The timing explanation is not supported.** The average responses do peak at
different times, but lining up the peaks does not make a borrowed seed useful. Whatever stops a seed from
travelling is more than a delay: most likely the shape of the response and how it sits relative to each
setup's background EEG, which the decoder weights heavily through the inverse covariance.

## UMM fidelity grid (descriptive)

Rome ALS, the authors' preprocessing, every variant the paper describes. Published figure: 77.86%.

| Covariance | Means | Accuracy | Participants at 10% or below |
|---|---|---:|---:|
| current letter | current letter | 49.3% | 0 |
| current letter | optimistic memory | 42.9% | 4 |
| current letter | confidence memory | 53.6% | 3 |
| pooled | current letter | 43.9% | 0 |
| pooled | optimistic memory | 48.2% | 4 |
| **pooled** | **confidence memory** | **69.6%** | 2 |

Centring each letter's epochs before pooling, as the authors' demo does, changed nothing or cost under a point.
The strongest variant is the one the main study already uses as `umm_memory`. No variant reaches the published
figure, so the gap is not a matter of picking the wrong configuration; it lies in something the paper and demo
do not specify for this dataset, or in the authors' unpublished implementation.

## What this leaves

- The main result stands unchanged: a seed from the same setup helps people with ALS.
- A seed from a different setup should not be used. Neither a label-free latency search (here) nor CORAL
  alignment ([ADAPTIVE_STUDY.md](ADAPTIVE_STUDY.md)) closes the gap.
- The practical route for a new clinic is therefore a local seed: labelled recordings from a handful of
  earlier users of that same setup. How few are enough was not tested.
