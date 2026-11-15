# Calibration-free P300 spelling for people with ALS

Research code and results. No EEG recordings are stored here; scripts download them from their public
sources. This is not a medical device and nothing here is evidence of clinical benefit
([DATA_NOTICE.md](DATA_NOTICE.md)).

## The question

A P300 speller lets a person pick letters with their brain response to flashes on a screen. Before it works,
the person normally has to sit through a calibration session, copying known letters for several minutes so
the decoder can learn their response. For someone with ALS that is tiring and has to be repeated.

UMM (Sosulski & Tangermann, 2023) is a published method that needs **no calibration**. It scores above 99%
on healthy users. On people with ALS its authors report 77.86%, with some participants at 0%, and they explain
why: after one early wrong guess, what the method remembers can point the wrong way, and every later letter
inherits the mistake.

**Does starting UMM's memory from other people's average brain response remove that failure, so that a new
user with ALS needs no calibration?**

## Answer in short

Yes, if the other people were recorded on the same kind of setup. No, and it can do harm, if they were not.

Tested on 19 people with ALS from two labs and 10 healthy people. Exact-letter accuracy, chance is 2.8%.
Brackets are 95% intervals over participants. Full tables: [RESULTS.md](RESULTS.md).

| All 19 people with ALS | Calibration from the user | Accuracy, all letters | Worst participant |
|---|---|---:|---:|
| Other people's data only, never adapting (`transfer`) | none | 58.7% (46.6-70.3) | 6.7% |
| UMM with memory, reimplemented (`umm_memory`) | none | 61.1% (43.8-77.9) | 0.0% |
| **Seeded UMM** (`seeded_umm`, seed from the same dataset) | **none** | **78.4%** (67.6-87.1) | 10.0% |

1. **Seeding helps people with ALS.** Seeded UMM beat UMM with memory by 17.3 points (95% CI +4.2 to +31.8;
   exact permutation test p = 0.027). This was the one test named in advance as the test of the main claim.
   It beat using other people's data without adapting by 19.7 points (p = 0.0001).
2. **The gain comes from rescuing failures, not from lifting everyone.** 8 participants improved, 5 were
   unchanged and 6 got slightly worse. UMM with memory left four people with ALS at 0-10%; seeded UMM left one.
3. **After the calibration period it is on par with a calibrated decoder.** On the letters after a normal
   calibration block, seeded UMM was 2.2 points below the calibrated decoder (95% CI -11.6 to +4.0). That
   interval is too wide to call them equal; it does show no calibration was needed to get into the same range.
4. **It replicated at a second site** (11 people with ALS, different lab, hardware and flashing pattern):
   72.4% vs 62.1% for UMM with memory. Alone, that difference is not statistically significant (p = 0.20).
5. **It does no harm where UMM already works.** On healthy users: 94.4% vs 91.7%.
6. **The seed does not travel between setups. This is the main limit.** Seeding Rome's ALS users from Rome's
   healthy users gave 73.9%, well short of the 86.8% from fellow patients. Seeding the second site's users
   from Rome's data gave 30.1%, far *worse* than no seed at all (62.1%). A new clinic could not simply reuse
   another lab's seed.
7. **It needs enough flashes.** With 3 or fewer flash sequences per letter it is worse than `transfer`; from
   5 sequences up it matches the calibrated decoder (Rome ALS data).

## Read this before quoting it

- **UMM is reimplemented here from the paper's equations.** The authors' algorithm file is not public (it is
  available from them on request and a patent is pending on UMM-based applications). With my preprocessing
  the reimplementation scores 59.6% on the Rome ALS data; with the settings from the authors' public demo it
  scores 69.6%; they report 77.86%. The target set in advance was to land within 5 points, and it did not. So
  the baseline here is weaker than the published one. Seeded UMM scores 86.8% and 87.9% on that dataset under
  the two pipelines, above the published figure too, but that comparison is across different code and
  preprocessing and is not a controlled one.
- **One participant at the second site is at about 10% under every calibration-free method**, and the
  calibrated decoder is also weakest at that site (82.1%).
- **19 people with ALS is still small**, and secondary comparisons are not corrected for multiple testing.
- **Offline replay.** Recorded sessions were replayed letter by letter. No one used the speller live.
- The seed is unit-free and its weight (five letters' worth) was fixed in advance, not tuned.
- Rules for what would count as a result were written before each run: [PROTOCOL.md](PROTOCOL.md), both
  addenda, including one deviation. Outcome against those rules:

| Set in advance | Outcome |
|---|---|
| Primary test: seeded UMM vs UMM with memory, 19 people with ALS, p < 0.05 | **Met** (p = 0.027) |
| Reimplemented UMM within 5 points of the published 77.86% under the authors' settings | **Not met** (69.6%) |
| Replicates at the second ALS site (higher mean than UMM with memory) | **Met** (72.4% vs 62.1%) |
| No more than 3 points worse than UMM with memory on healthy users | **Met** (2.8 points better) |
| Seed from another dataset beats UMM with memory and is within 5 points of the local seed | **Not met** in any of the three cases |

## Run it

Windows Command Prompt, from this folder. Python 3.10 or newer, about 800 MB of downloads, no paid services.
The full run takes about ten minutes.

```bash
py -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe download_nemar_source.py
```

```bash
.venv\Scripts\python.exe download_bnci009.py
```

```bash
.venv\Scripts\python.exe download_dataset.py
```

```bash
.venv\Scripts\python.exe calibration_free.py
```

```bash
.venv\Scripts\python.exe render_results.py
```

To verify that the checked-in report still matches the JSON (the same check runs in CI):

```bash
.venv\Scripts\python.exe render_results.py --check
```

```bash
.venv\Scripts\python.exe -m pytest -q
```

## What is in here

| File | What it does |
|---|---|
| `calibration_free.py` | Loads the three datasets, runs every method as an online replay, writes `results/calibration_free_results.json`. |
| `render_results.py` | Turns that JSON into [RESULTS.md](RESULTS.md), or checks it with `--check`. |
| `tests/` | Unit tests on synthetic spellers; no download needed. |
| `download_*.py` | Fetch the three public datasets. |
| [PROTOCOL.md](PROTOCOL.md) | What was fixed in advance, and deviations. |
| [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) | Prior work and the gap. |
| `exploratory/` | Earlier target-detection baselines. Not part of the result; see its README. |

## Datasets

| Name in the code | Source | People | Flashing pattern |
|---|---|---|---|
| `rome_als` | [BNCI 2014-008](https://www.nemar.org/dataset/nm000169) (Riccio et al., 2013) | 8 with ALS | row/column |
| `rome_healthy` | [BNCI 2014-009](https://bnci-horizon-2020.eu/database/data-sets) (Aricò et al., 2014) | 10 healthy | row/column |
| `etsu_als` | [bigP3BCI](https://doi.org/10.13026/0byy-ry86) Study L, checkerboard training files | 11 with ALS | checkerboard |

All are 6x6 spellers at 256 Hz, cut down to the eight electrodes they share.

## What would make this stronger

- The authors' own UMM code, to replace the reimplemented baseline.
- A live test, where the user sees feedback and the early, less accurate letters matter.
- Finding out *why* the seed fails across setups (flash timing, amplifier filters, flashing pattern) and
  whether aligning the recordings first repairs it.
- More people with ALS; bigP3BCI has three further studies that include them.

Code: MIT ([LICENSE](LICENSE)). Data licences differ by dataset; see [DATA_NOTICE.md](DATA_NOTICE.md).
