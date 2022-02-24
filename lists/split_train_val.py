import os
import argparse
import pandas as pd
from sklearn import model_selection
from src.utils.random_seed import SEED

SAVE_DIR = './lists/'
DATA_PATH = './data/train.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_frac', default=.2)
    parser.add_argument('--seed', default=SEED)

    args = parser.parse_args()

    test_frac = float(args.test_frac)
    seed = int(args.seed)

    df = pd.read_csv(DATA_PATH)
    train_df, val_df = model_selection.train_test_split(
        df, test_size=test_frac, random_state=seed, stratify=df.labels.values
    ) 
    train_df.to_csv(f'{SAVE_DIR}train.csv', index=False)
    val_df.to_csv(f'{SAVE_DIR}val.csv', index=False)