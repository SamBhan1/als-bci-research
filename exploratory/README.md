# Earlier exploratory work

These scripts came before the calibration-free study in the repository root and are kept for the record.
**They are not part of the main result and their numbers should not be cited.**

Why they are set aside:

- Most of them score *target vs non-target detection* on averaged epochs (balanced accuracy, ROC-AUC). That is
  not spelling accuracy, and the averaging mixes epochs in a way a live speller could not.
- `source_character_eval.py` had a letter-boundary error (the first flash of each letter was assigned to the
  previous letter). It was fixed on 15 Nov 2026. Re-running the xDAWN + shrinkage-LDA spelling evaluation after
  the fix gave 0.911 at 80% calibration (56 held-out letters) and 0.969 at 90% (32 held-out letters). Other
  spelling numbers in `EARLIER_RESULTS.md` (0.725, 0.734, 0.844, 0.906) were computed before the fix and have
  not been re-run.
- Calibration fractions of 60-90% leave very few held-out letters per person; those results are exploratory.
- The checked-in protocol for these scripts was written alongside the runs, not before them.

`EARLIER_RESULTS.md` is the old results file under a new name. Its first section is the first calibration-free run (eight participants, seed in raw units), superseded by `RESULTS.md` in the root.

The scripts import each other, so run them from inside this folder, for example:

```cmd
cd exploratory
..\.venv\Scripts\python.exe source_character_eval.py --source-dir ..\data\nm000169\sourcedata --training-source calibration --calibration-fraction 0.8 --repeats 10 --model xdawn_lda
```

`download_nemar.py` fetches the BIDS-format copy of the ALS dataset that `external_validation.py` uses; the
main study uses the source-format files from `download_nemar_source.py` in the root instead.
