import argparse
import io
import pickle
import textwrap

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None
def _wrap(label, value, width=100):
    wrapped = textwrap.fill(str(value), width=width, subsequent_indent="  ")
    return f"{label}\n  {wrapped}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="Path to the pickle file")
    parser.add_argument("--map_location", type=str, help="Device location", default='cpu')
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
    if not data:
        raise ValueError("Pickle file is empty or contains no examples.")

    first = data[0]

    print(f"Loaded {len(data)} examples")
    print("First example fields and values:")
    if isinstance(first, dict):
        for key in sorted(first.keys()):
            print(_wrap(f"- {key}:", first.get(key)))
    else:
        print(_wrap("Value:", first))


if __name__ == "__main__":
    main()

