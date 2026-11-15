# Data and usage notice

This repository contains code and documentation only. Raw EEG files are intentionally excluded.

The experiments use public datasets downloaded by the included scripts:

- PhysioNet bigP3BCI: https://www.physionet.org/content/bigp3bci/1.0.0/
- NEMAR BNCI 2014-008 ALS P300: https://www.nemar.org/dataset/nm000169
- BNCI 2014-009 (ten healthy users, same lab as 2014-008): https://bnci-horizon-2020.eu/database/data-sets

The source-format NEMAR `.mat` files are used only locally to recover row/column stimulus codes for spelling-level evaluation; they are also covered by the NEMAR license.

Users must read and follow each dataset's current license, attribution, and access terms. The NEMAR dataset is CC BY-NC-ND 4.0 and must not be redistributed or used to distribute derivative files or model artifacts without permission.

This is research software, not a medical device. It has not been clinically validated and must not be used for diagnosis, treatment, or patient communication without appropriate ethics, safety, regulatory, and clinical review.


`results/calibration_free_results.json` holds accuracy figures per participant and nothing else: no EEG, no
model weights. The seed vectors computed from other participants are built at run time and are not saved.

UMM (Sosulski & Tangermann, 2023) is reimplemented in `calibration_free.py` from the equations in the paper
for research comparison. The authors' own implementation is distributed only on request, and they state that a
patent is pending on UMM-based applications. Anyone planning to use UMM, or the seeded variant here, outside
research should contact the authors first: https://github.com/jsosulski/umm_demo
