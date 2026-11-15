"""Download BNCI 2014-009 (ten healthy users, 6x6 row/column P300 speller, same lab as the ALS set).

About 185 MB into data/bnci009/. Dataset page: https://bnci-horizon-2020.eu/database/data-sets
Reference: Arico et al. (2014), "Influence of P300 latency jitter on event related potential-based
brain-computer interface performance", J. Neural Eng. 11(3):035008.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://lampx.tugraz.at/~bci/database/009-2014/A{subject:02d}S.mat"


def main() -> int:
    out = Path(__file__).parent / "data" / "bnci009"
    out.mkdir(parents=True, exist_ok=True)
    for subject in range(1, 11):
        target = out / f"A{subject:02d}S.mat"
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"{target.name} already present")
            continue
        url = BASE.format(subject=subject)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
