import os

import pandas as pd
import plac
from sklearn.model_selection import train_test_split


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    out_path=("Path to save split data", "option", "o", str)
)
def main(data_path='data/iris.csv', out_path='data/split/'):
    df = pd.read_csv(data_path)
    train, test = train_test_split(df, stratify=df['class'].values, test_size=0.2, random_state=42)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    train.to_csv(f'{out_path}train.csv', index=False)
    test.to_csv(f'{out_path}test.csv', index=False)
    print(f'Finished Splitting Data:\nStats:')
    print(f'\tTotal: {df.shape}\tClass vise samples: {df["class"].value_counts().values}')
    print(f'\tTrain: {train.shape}\tClass vise samples: {train["class"].value_counts().values}')
    print(f'\tTest: {test.shape}\tClass vise samples: {test["class"].value_counts().values}')


if __name__ == '__main__':
    plac.call(main)
