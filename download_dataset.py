"""Download a small, reproducible Study L subset from PhysioNet."""

from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve


BASE = "https://physionet.org/files/bigp3bci/1.0.0/bigP3BCI-data/StudyL"
OUT = Path(__file__).resolve().parent / "data" / "raw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for number in range(1, 12):
        subject = f"L_{number:02d}"
        session_url = f"{BASE}/{subject}/SE001/Train/CB/"
        for recording in range(1, 6):
            filename = f"{subject}_SE001_CB_Train{recording:02d}.edf"
            destination = OUT / filename
            if destination.exists() and destination.stat().st_size > 0:
                print(f"exists {filename}")
                continue
            try:
                print(f"downloading {filename}")
                urlretrieve(session_url + filename, destination)
            except HTTPError:
                print(f"unavailable {filename}")
    print(f"Downloaded EDF files to {OUT}")


if __name__ == "__main__":
    main()
