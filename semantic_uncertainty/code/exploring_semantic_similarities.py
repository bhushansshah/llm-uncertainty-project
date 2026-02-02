import argparse
import csv
import io
import pickle
import textwrap
import torch


def _wrap(label, value, width=100):
    wrapped = textwrap.fill(str(value), width=width, subsequent_indent="  ")
    return f"{label}\n  {wrapped}"


def load_pickle(filepath, map_location='cpu'):
    """Load pickle file with torch tensor support."""
    if torch is None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    else:
        device = torch.device(map_location)
        original_load_from_bytes = torch.storage._load_from_bytes

        def _load_from_bytes_cpu(b):
            return torch.load(io.BytesIO(b), map_location=device, weights_only=False)

        torch.storage._load_from_bytes = _load_from_bytes_cpu
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        finally:
            torch.storage._load_from_bytes = original_load_from_bytes
    return data


def main():
    parser = argparse.ArgumentParser(description="Explore semantic similarities outputs")
    parser.add_argument("--csv_filepath", type=str, help="Path to deberta_predictions CSV file")
    parser.add_argument("--pkl_filepath", type=str, help="Path to generations_similarities pickle file")
    parser.add_argument("--question_id", type=str, help="Question ID to look up in the pickle file")
    parser.add_argument("--num_csv_entries", type=int, default=3, help="Number of CSV entries to display")
    parser.add_argument("--map_location", type=str, help="Device location", default='cpu')
    args = parser.parse_args()

    # Display entries from CSV file
    if args.csv_filepath:
        print("=" * 80)
        print(f"DEBERTA PREDICTIONS CSV - First {args.num_csv_entries} entries")
        print("=" * 80)
        with open(args.csv_filepath, 'r', encoding='UTF8') as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Columns: {header}")
            print("-" * 80)
            for i, row in enumerate(reader):
                if i >= args.num_csv_entries:
                    break
                print(f"\nEntry {i + 1}:")
                for col_name, value in zip(header, row):
                    print(_wrap(f"  {col_name}:", value))
        print()

    # Display sample from pickle file
    if args.pkl_filepath and args.question_id:
        print("=" * 80)
        print(f"GENERATIONS SIMILARITIES PKL - Question ID: {args.question_id}")
        print("=" * 80)
        data = load_pickle(args.pkl_filepath, args.map_location)

        print(f"Total entries in pickle file: {len(data)}")
        print("-" * 80)

        if args.question_id in data:
            args.question_id = list(data.keys())[24]
            sample = data[args.question_id]
            print(f"\nSample for question ID '{args.question_id}':")
            if isinstance(sample, dict):
                for key in sorted(sample.keys()):
                    print(_wrap(f"  {key}:", sample.get(key)))
            else:
                print(_wrap("  Value:", sample))
        else:
            print(f"\nQuestion ID '{args.question_id}' not found in pickle file.")
            print("Available keys (first 10):")
            for i, key in enumerate(list(data.keys())[:10]):
                print(f"  - {key}")
            if len(data) > 10:
                print(f"  ... and {len(data) - 10} more")


if __name__ == "__main__":
    main()

