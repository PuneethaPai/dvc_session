import os

import pandas as pd
import plac
from sklearn.decomposition import PCA


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    out_path=("Path to save featurized data", "option", "o", str)
)
def main(data_path='data/split/', out_path='data/features/'):
    train = pd.read_csv(f'{data_path}train.csv')
    test = pd.read_csv(f'{data_path}test.csv')

    source_features = train.drop(columns=['class'])
    pca = PCA(n_components=2).fit(source_features)
    train_feature = pd.DataFrame(pca.transform(source_features))
    test_feature = pd.DataFrame(pca.transform(test.drop(columns=['class'])))

    train_feature['class'] = train['class']
    test_feature['class'] = test['class']
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    train_feature.to_csv(f'{out_path}train.csv', index=False)
    test_feature.to_csv(f'{out_path}test.csv', index=False)
    print(f'Finished Feature Engineering:\nStats:')
    print(f'\tExplained Variance: {pca.explained_variance_}')
    print(f'\tExplained Variance Ratio: {pca.explained_variance_ratio_}')


if __name__ == '__main__':
    plac.call(main)
