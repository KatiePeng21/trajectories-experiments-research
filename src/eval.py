# src/eval.py
import argparse, json, csv, os, sys

def main():
    p = argparse.ArgumentParser(description="Minimal eval: dump metrics.json -> CSV")
    p.add_argument("--config", required=True, help="Path to experiment config (unused here, kept for CLI parity)")
    p.add_argument("--checkpoint", help="Path to checkpoint (unused in this minimal stub)")
    p.add_argument("--save_csv", required=True, help="Where to write CSV (e.g., experiments/.../eval_test.csv)")
    p.add_argument("--metrics_json", help="Optional explicit path to metrics.json")
    p.add_argument("--split", default="test", help="Split label to include in CSV (default: test)")
    args = p.parse_args()

    # infer metrics.json alongside config unless explicitly provided
    metrics_path = args.metrics_json
    if metrics_path is None:
        # assume config like experiments/exp_XXX/config.yaml
        exp_dir = os.path.dirname(os.path.abspath(args.config))
        metrics_path = os.path.join(exp_dir, "metrics.json")

    if not os.path.isfile(metrics_path):
        print(f"[eval] metrics.json not found at: {metrics_path}", file=sys.stderr)
        # still create an empty CSV header so the file isn't zero-byte
        _write_csv(args.save_csv, {"split": args.split})
        sys.exit(1)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # flatten + tag with split
    out = {"split": args.split, **metrics}

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    _write_csv(args.save_csv, out)
    print(f"[eval] wrote CSV -> {args.save_csv}")
    for k, v in out.items():
        print(f"  {k}: {v}")

def _write_csv(path, rowdict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(rowdict.keys())
        w.writerow(rowdict.values())

if __name__ == "__main__":
    main()
