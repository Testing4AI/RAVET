import argparse
from pathlib import Path
from datasets import load_dataset

from sentinelrag.utils.paths import default_datasets_dir


DEFAULT_OUTPUT_DIR = default_datasets_dir()

def main() -> None:
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face.")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="The name of the dataset to download (e.g., 'yixuantt/MultiHopRAG').",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="The configuration name of the dataset (optional).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="The split of the dataset to download (optional).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to store datasets (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--save-name",
        default=None,
        help="Name of the directory to save the dataset to. Defaults to dataset-name (slashes replaced by underscores).",
    )
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    
    # Determine save name
    if args.save_name:
        save_name = args.save_name
    else:
        # Create a default save name from dataset name and config
        save_name = args.dataset_name.replace("/", "_")
        if args.config_name:
            save_name += f"_{args.config_name}"

    dataset_dir = output_dir / save_name
    
    print(f"Downloading {args.dataset_name} (config: {args.config_name}, split: {args.split}) into {dataset_dir}...")
    
    # Load the dataset
    load_kwargs = {}
    if args.config_name:
        load_kwargs["name"] = args.config_name
    if args.split:
        load_kwargs["split"] = args.split
    ds = load_dataset(args.dataset_name, **load_kwargs)
    
    # Save to disk
    ds.save_to_disk(str(dataset_dir))
    print(f"[done] Saved {args.dataset_name} to {dataset_dir}")

if __name__ == "__main__":
    main()
