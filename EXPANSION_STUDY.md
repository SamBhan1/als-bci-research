# Expansion: 27 more people with ALS

Produced by `expansion_study.py`; numbers are in `results/expansion_study_results.json`. The plan, including
what would count as passing, was written and pushed before the new recordings were opened:
[PROTOCOL.md](PROTOCOL.md), addendum 4. Exact-letter accuracy, participant mean, 95% bootstrap interval over
participants. Chance is 2.8%.

New data: bigP3BCI Study B (19 people with ALS) and Study N (8 people with ALS, each recorded once with a wet
cap and once with a dry cap). Same lab as the earlier 11-person Study L, 6x6 checkerboard speller.

## A. Does the main result hold on new people? Not convincingly.

This was the pre-specified primary test: the 27 people who played no part in developing the method
(Study B and Study N wet cap), seeded UMM vs UMM with memory, all letters.

| | Seeded UMM | UMM with memory | Difference | Better / same / worse | p (sign-flip) |
|---|---:|---:|---:|---|---:|
| **27 new people (primary)** | 83.0% (71.5-92.5) | 76.9% | **+6.1** (-2.2 to +15.9) | 7 / 13 / 7 | **0.21** |
| All 46 people with ALS | 81.1% (73.4-88.0) | 70.4% | +10.8 (+3.3 to +18.9) | 15 / 18 / 13 | 0.011 |

- **The primary test failed.** The direction is the same as before, but the gain is a third of the size and
  could be zero.
- The 46-person figure is significant, but it includes the 19 people the method was developed on, so it is
  **not** an independent confirmation. It is reported because the protocol said it would be.
- Why the gain shrank: in Study B plain UMM with memory already works well (77.0%) and has fewer collapses
  to 0%, so there is less for the seed to rescue. Seeding still beats *never adapting* by a wide margin
  (+14.8 points in Study B, p = 0.0005; +20.8 in Study N wet, p = 0.031).

| New dataset | Transfer only | UMM with memory | Seeded UMM | Calibrated decoder (after 12 letters) |
|---|---:|---:|---:|---:|
| Study B, 19 people | 65.3% | 77.0% | 80.1% | 86.9% |
| Study N wet cap, 8 people | 69.2% | 76.7% | 90.0% | 97.2% |
| Study N dry cap, same 8 people | 7.1% | 20.0% | 28.8% | 38.2% |

In Study B seeded UMM is 5.1 points below the calibrated decoder after its calibration letters
(-12.2 to +0.5). The earlier "on par with a calibrated decoder" result does not clearly carry over.

## B. How few earlier users make a working seed? About 12, and small seeds hurt.

Study B, each person seeded from k other people chosen at random (10 draws each).

| Donors | 1 | 2 | 3 | 5 | 8 | 12 | 18 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seeded UMM | 58.7% | 65.3% | 69.6% | 72.9% | 76.7% | **78.5%** | 80.1% |

UMM with memory, no seed: 77.0%. **A seed from fewer than about 8 people is worse than no seed at all.** By
the rule set in advance (above no seed and within 3 points of the 18-person seed), 12 donors is the smallest
that is enough. A new clinic would need labelled sessions from roughly a dozen earlier users of its own setup.

## C. Does a seed travel within one lab? Mostly yes; between labs, almost never.

A borrowed seed "travels" if it beats no seed and is within 5 points of the local seed.

| Between | Ordered pairs | Seeds that travel | Average gain over no seed | Average shortfall vs local seed |
|---|---:|---:|---:|---:|
| Studies of the same lab (L, B, N wet) | 6 | **4** | +2.6 points | -6.2 points |
| Different labs, people with ALS only | 10 | **1** | -6.4 points | -20.3 points |

Study B and Study L seed each other almost as well as a local seed (82.5% and 67.9%, local 80.1% and 72.3%).
This supports the explanation that **the obstacle is the recording setup, not the people**: other people with
ALS help if they were recorded on the same equipment and paradigm.

## D. Wet vs dry caps (descriptive)

With the dry cap, every method is close to unusable for these eight people, including the calibrated decoder
(38.2% after 12 letters, against 97.2% with the wet cap). The dry-cap recordings carry much more noise (their
overall amplitude is 2-5 times the wet-cap recordings). Seeding the dry sessions from the same people's wet
sessions gave 33.3%: the best of any method there, but still far from usable. This is a recording-quality
limit, not something a decoder fixes.

## E. What predicts whether a borrowed seed helps? Not the measure I predicted.

Across 28 ordered pairs of datasets (wet/dry pairs of the same people excluded), Spearman correlation between
how alike two datasets' average responses are and how much the borrowed seed helps:

| Similarity measure | rho | p |
|---|---:|---:|
| Plain cosine | 0.46 | 0.015 |
| Adjusted for the target's background EEG (pre-specified) | 0.14 | 0.48 |

The pre-specified explanation (similarity *as the decoder sees it*, background EEG divided out) is **not
supported**: it does not predict which seeds help. Plain similarity of the response shape predicts it
moderately. So what matters is mostly whether the average response looks alike, and the background-noise
correction added nothing.

## Against the rules set in advance

| Set in advance | Outcome |
|---|---|
| A. Seeded UMM beats UMM with memory on the 27 new people, p < 0.05 | **Not met** (+6.1 points, p = 0.21) |
| B. Smallest seed that is above no seed and within 3 points of the full seed | 12 people |
| C. Seeds travel within one lab but not between labs | **Supported** (4 of 6 vs 1 of 10) |
| E. Background-adjusted similarity predicts the gain (rho at least 0.6, and above plain cosine) | **Not met** (0.14 vs 0.46) |

## Limits

- All 27 new people come from one lab. Studies B, L and N may share participants; the files do not allow
  checking, so the within-lab results in C may be optimistic.
- Study B letters have 180 flashes each (15 sequences), more than the other datasets.
- 28 pairs share datasets, so the correlations in E are not 28 independent observations.
- Offline replay, as throughout.
