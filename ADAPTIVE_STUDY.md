# Adaptive extensions: cross-site alignment and confidence-based stopping

This is an exploratory extension of the primary calibration-free study. It does not
replace the headline result in [RESULTS.md](RESULTS.md).

## What was added

`adaptive_study.py` tests two label-free adaptations:

1. **CORAL seed alignment.** Source participants are feature-standardised using their
   own recordings. For a new participant, the seed is transported into the new
   participant's centre, scale, and covariance geometry using only flashes observed
   so far. The test participant's target labels are never used.
2. **Confidence-based stopping.** After each sequence, UMM can stop when its confidence
   reaches a fixed threshold. Otherwise it collects another sequence; if the threshold
   is never reached, it records a forced decision at the maximum. The default is a
   threshold of 1.0 after at least two sequences. The threshold is exploratory, not a
   clinically tuned operating point.

The sequence lengths are 12 flashes for the Rome row/column recordings and 18 for the
ETSU checkerboard recordings. All evaluations use the primary binned pipeline and the
same participant-disjoint rules as the main study.

Run it with:

```cmd
.venv\Scripts\python.exe adaptive_study.py --data-dir data --thresholds 1.0 --min-sequences 2
```

## Preliminary result on the downloaded recordings

The JSON below was generated on 15 November 2026 from the three local datasets. Accuracy
is exact-letter participant mean; intervals are participant bootstrap intervals. The
cross-site comparisons are secondary and are not multiplicity-corrected.

| Target group | Seed source | Raw seeded UMM | CORAL seeded UMM | Difference |
|---|---|---:|---:|---:|
| Rome ALS | other Rome ALS participants | 86.8% | **90.4%** | +3.6 points |
| ETSU ALS | other ETSU ALS participants | 72.4% | **74.9%** | +2.6 points |
| Rome ALS | Rome healthy participants | 73.9% | **83.6%** | +9.6 points |
| Rome ALS | ETSU ALS participants | 64.6% | 61.8% | −2.9 points |
| ETSU ALS | Rome ALS participants | 30.1% | 38.0% | +7.8 points |
| ETSU ALS | Rome healthy participants | 40.1% | 46.2% | +6.1 points |

CORAL improves several transfers, but it does **not** solve cross-site generalisation:
the best cross-site ETSU result is still far below the 74.9% local-seed result. The
Rome-ALS local comparison is also not statistically conclusive at n=8 (exact sign-flip
p=0.1875). These are hypotheses for a larger, pre-registered study, not a clinical
performance claim.

## Confidence stopping result

On the eight Rome ALS participants, with a fixed threshold of 1.0 and a minimum of two
sequences:

| Method | Accuracy | Mean sequences per letter | Forced at maximum |
|---|---:|---:|---:|
| Raw seeded UMM | 82.9% | 7.22 | 41.1% |
| CORAL seeded UMM | **87.9%** | **6.98** | 37.5% |

The stopping result is exploratory. It trades some decisions before the full ten
sequences for a modest reduction in flashes; many letters still reach the maximum.
Future work should select the threshold on training participants and evaluate it on
held-out participants, then report accuracy, error rate, flashes, time, and abstentions
together.
