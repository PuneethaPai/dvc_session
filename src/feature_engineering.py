import os
import yaml
from yaml import CLoader as Loader

import pandas as pd
import plac
from sklearn.decomposition import PCA

from utils import log_experiment, save_results, read_data


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    feature_path=("Path to save featurized data", "option", "f", str),
    out_path=("Path to save pca model", "option", "o", str)
)
def main(data_path='data/split/', feature_path='data/features/', out_path='data/pca/'):
    X_train, X_test, y_train, y_test = read_data(data_path)

    params = read_params('params.yaml', 'pca')
    pca = PCA(**params).fit(X_train)

    train_feature = pd.DataFrame(pca.transform(X_train))
    test_feature = pd.DataFrame(pca.transform(X_test))
    train_feature['class'] = y_train
    test_feature['class'] = y_test

    if not os.path.isdir(feature_path):
        os.mkdir(feature_path)
    train_feature.to_csv(f'{feature_path}train.csv', index=False)
    test_feature.to_csv(f'{feature_path}test.csv', index=False)
    save_results(out_path, pca, None)

    print(f'Finished Feature Engineering:\nStats:')
    print(f'\tExplained Variance: {pca.explained_variance_}')
    print(f'\tExplained Variance Ratio: {pca.explained_variance_ratio_}')

    log_experiment(out_path, params=params,
                   metrics=dict(explained_variance_=pca.explained_variance_,
                                explained_variance_ratio_=pca.explained_variance_ratio_))


def read_params(file='params.yaml', model='pca'):
    with open(file, 'r') as fp:
        params = yaml.load(fp, Loader)
    return params[model]


if __name__ == '__main__':
    plac.call(main)
