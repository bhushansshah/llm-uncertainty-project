import pickle
import argparse
import io

try:
    import torch
except Exception:
    torch = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, required=True, help="Path to the pickle file")
    parser.add_argument("--map_location", type=str, default='cpu', help="Device location")
    args = parser.parse_args()

    if torch is None:
        with open(args.filepath, "rb") as f:
            data = pickle.load(f)
    else:
        map_location = torch.device(args.map_location)
        original_load_from_bytes = torch.storage._load_from_bytes

        def _load_from_bytes_cpu(b):
            return torch.load(io.BytesIO(b), map_location=map_location, weights_only=False)

        torch.storage._load_from_bytes = _load_from_bytes_cpu
        try:
            with open(args.filepath, "rb") as f:
                data = pickle.load(f)
        finally:
            torch.storage._load_from_bytes = original_load_from_bytes

    # Print first example for specific fields
    print("First example:")
    print(f"  id: {data['ids'][0]}")
    print(f"  number_of_semantic_sets: {data['number_of_semantic_sets'][0]}")
    print(f"  predictive_entropy_over_concepts: {data['predictive_entropy_over_concepts'][0]}")
    print(f"  unnormalised_entropy_over_concepts: {data['unnormalised_entropy_over_concepts'][0]}")


if __name__ == "__main__":
    main()
