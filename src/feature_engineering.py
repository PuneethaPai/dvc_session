import os

import pandas as pd
import plac
from sklearn.decomposition import PCA


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    out_path=("Path to save split data", "option", "o", str)
)
def main(data_path='data/split/', out_path='data/features/'):
    train = pd.read_csv(f'{data_path}train.csv')
    test = pd.read_csv(f'{data_path}test.csv')
    train.drop(columns=['class'], inplace=True)
    test.drop(columns=['class'], inplace=True)

    pca = PCA(n_components=2).fit(train)
    train_feature = pd.DataFrame(pca.transform(train))
    test_feature = pd.DataFrame(pca.transform(test))
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    train_feature.to_csv(f'{out_path}train.csv', index=False)
    test_feature.to_csv(f'{out_path}test.csv', index=False)
    print(f'Finished Feature Engineering:\nStats:')
    print(f'\tExplained Variance: {pca.explained_variance_}')
    print(f'\tExplained Variance Ratio: {pca.explained_variance_ratio_}')


if __name__ == '__main__':
    plac.call(main)
