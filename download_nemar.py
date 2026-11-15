"""Download the eight-participant NEMAR ALS P300 external-validation set."""

from pathlib import Path
from urllib.request import urlretrieve


BASE = "https://data.nemar.org/nm000169/v1.0.3"
OUT = Path(__file__).resolve().parent / "data" / "nm000169"


def main() -> None:
    for number in range(1, 9):
        subject = f"sub-{number}"
        folder = OUT / subject / "ses-0" / "eeg"
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{subject}_ses-0_task-p300_run-0_"
        for suffix in ("eeg.bdf", "events.tsv", "channels.tsv", "eeg.json"):
            filename = prefix + suffix
            destination = folder / filename
            if destination.exists() and destination.stat().st_size > 0:
                print(f"exists {filename}")
                continue
            url = f"{BASE}/{subject}/ses-0/eeg/{filename}"
            print(f"downloading {filename}")
            urlretrieve(url, destination)
    print(f"Downloaded NEMAR ALS P300 data to {OUT}")


if __name__ == "__main__":
    main()
