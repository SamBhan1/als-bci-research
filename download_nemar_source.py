"""Download NEMAR's source-format ALS P300 recordings with stimulus codes."""

from pathlib import Path
from urllib.request import urlretrieve


BASE = "https://data.nemar.org/nm000169/v1.0.3/sourcedata"
OUT = Path(__file__).resolve().parent / "data" / "nm000169" / "sourcedata"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for number in range(1, 9):
        filename = f"A{number:02d}.mat"
        destination = OUT / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"exists {filename}")
            continue
        print(f"downloading {filename}")
        urlretrieve(f"{BASE}/{filename}", destination)
    print(f"Downloaded source recordings to {OUT}")


if __name__ == "__main__":
    main()
