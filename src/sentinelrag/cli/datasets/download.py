import argparse
from pathlib import Path

from beir import util

from sentinelrag.utils.paths import default_datasets_dir

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DEFAULT_OUTPUT_DIR = default_datasets_dir()
SUPPORTED_DATASETS = {"nfcorpus", "msmarco", "msmarco_200k", "trec-covid", "hotpotqa", "nq"}


def download_dataset(dataset: str, output_dir: Path) -> Path:
    """Download and unpack a single BEIR dataset into output_dir/dataset."""
    dataset = dataset.lower()
    # if SUPPORTED_DATASETS and dataset not in SUPPORTED_DATASETS:
    #     raise ValueError(f"Unsupported dataset '{dataset}'. Choose from {sorted(SUPPORTED_DATASETS)}.")

    target_dir = output_dir / dataset
    if target_dir.exists():
        print(f"[skip] {dataset} already present at {target_dir}")
        return target_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{dataset}.zip"
    print(f"[download] Fetching {dataset} from {url}")
    data_path = Path(util.download_and_unzip(url, str(output_dir)))
    print(f"[done] Extracted to {data_path}")
    return data_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BEIR datasets into datasets/<name> directories.")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["nfcorpus"],
        help=f"Dataset names to download (default: %(default)s). Supported: {', '.join(sorted(SUPPORTED_DATASETS))}.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to store datasets (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    for dataset in args.datasets:
        download_dataset(dataset, output_dir)


if __name__ == "__main__":
    main()
