"""Self-provisioning data download.

The Kaggle credit-card fraud dataset is ~98MB, too large to commit to GitHub
cleanly, so it is gitignored and fetched on first run from a raw GitHub mirror.
Both train.py and app.py call ensure_data() before touching the CSV.
"""
from pathlib import Path
import urllib.request
import sys

DATA_URL = (
    "https://raw.githubusercontent.com/nsethi31/"
    "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
)
DATA_PATH = Path(__file__).parent / "data" / "creditcard.csv"
EXPECTED_ROWS = 284807  # data rows, excluding header


def ensure_data(path: Path = DATA_PATH) -> Path:
    """Download the dataset to `path` if it is missing. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 50_000_000:
        return path
    print(f"Downloading dataset to {path} ...", file=sys.stderr)
    urllib.request.urlretrieve(DATA_URL, path)
    print("Download complete.", file=sys.stderr)
    return path


if __name__ == "__main__":
    ensure_data()
