"""Download two more bigP3BCI studies whose participants all have ALS (6x6 checkerboard spellers).

  Study B  19 people, first session, training files      -> data/studyB/
  Study N   8 people, two sessions (wet and dry caps)     -> data/studyN/

About 1.2 GB. Dataset: https://doi.org/10.13026/0byy-ry86 (CC BY 4.0).
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/bigp3bci/1.0.0/bigP3BCI-data"
STUDY_B = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21]
STUDY_N = range(1, 9)


def listing(url: str, pattern: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return sorted(set(re.findall(pattern, response.read().decode("utf-8", "replace"))))


def fetch(folder_url: str, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in listing(folder_url, r'href="([^"?/]+\.edf)"'):
        target = out / name
        if not (target.exists() and target.stat().st_size > 100_000):
            print(f"downloading {name}", flush=True)
            urllib.request.urlretrieve(folder_url + name, target)
        count += 1
    return count


def main() -> int:
    root = Path(__file__).parent / "data"
    total = 0
    for number in STUDY_B:
        total += fetch(f"{BASE}/StudyB/B_{number:02d}/SE001/Train/CB/", root / "studyB")
    for number in STUDY_N:
        for session in ("SE001", "SE002"):
            session_url = f"{BASE}/StudyN/N_{number:02d}/{session}/Train/"
            for condition in listing(session_url, r'href="([A-Za-z]+)/"'):
                total += fetch(f"{session_url}{condition}/", root / "studyN")
    print(f"{total} files present under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
