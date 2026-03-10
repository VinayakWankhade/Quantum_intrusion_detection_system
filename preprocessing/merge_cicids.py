import pandas as pd
import os

DATA_PATH = "data/raw"
OUTPUT_PATH = "data/processed/cicids_merged.csv"

def merge_cicids():

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".parquet")]

    dataframes = []

    for file in files:

        print("Loading:", file)

        df = pd.read_parquet(os.path.join(DATA_PATH, file))

        dataframes.append(df)

    merged = pd.concat(dataframes, ignore_index=True)

    print("Merged shape:", merged.shape)

    merged.to_csv(OUTPUT_PATH, index=False)

    print("Saved dataset:", OUTPUT_PATH)


if __name__ == "__main__":
    merge_cicids()