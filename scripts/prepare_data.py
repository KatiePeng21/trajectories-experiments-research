# scripts/prepare_data.py
import pandas as pd
import glob
import os

def combine_csvs(input_dir, output_file):
    files = sorted(glob.glob(os.path.join(input_dir, "TRAJ_*.csv")))
    if not files:
        print(f"No TRAJ_*.csv files found in {input_dir}")
        return

    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df["traj_id"] = i + 1  # keep track of which trajectory this row came from
        dfs.append(df)

    big_df = pd.concat(dfs, ignore_index=True)
    big_df.to_parquet(output_file, index=False)
    print(f"Combined {len(files)} CSVs with {len(big_df)} rows -> {output_file}")

if __name__ == "__main__":
    input_dir = "data/processed/Oslo"
    output_file = "data/processed/oslo.parquet"
    combine_csvs(input_dir, output_file)
