import _pickle as cpickle
import os

import matplotlib.pyplot as plt
import pandas as pd
import plac
from dagshub import dagshub_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    n_estimators=("Path to save trained Model", "option", "e", str),
    max_samples=("Path to save trained Model", "option", "s", str),
    out_path=("Path to save trained Model", "option", "o", str)
)
def main(data_path='data/features/', out_path='data/models/r_forrest/', n_estimators=10, max_samples=30):
    train = pd.read_csv(f'{data_path}train.csv')
    test = pd.read_csv(f'{data_path}test.csv')

    X_train, y_train = train.drop(columns=['class']), train['class']
    X_test, y_test = test.drop(columns=['class']), test['class']

    model = RandomForestClassifier(n_estimators=n_estimators, max_samples=max_samples, n_jobs=4)
    model.fit(X_train, y_train)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open(f'{out_path}model.pkl', 'wb+') as fp:
        cpickle.dump(model, fp)

    cmd = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Reds)
    cmd.figure_.savefig(f'{out_path}confusion_matrix.svg', format='svg')
    c_matrix = cmd.confusion_matrix
    accuracy = model.score(X_test, y_test)

    print(f'Finished Training RandomForrest Model:\nStats:')
    print(f'\tConfusion Matrix:\n{c_matrix}')
    print(f'\tModel Accuracy: {accuracy}')
    with dagshub_logger(metrics_path=f'{out_path}metrics.csv', hparams_path=f'{out_path}params.yml') as logger:
        logger.log_hyperparams(model.get_params())
        logger.log_metrics(accuracy=accuracy)


if __name__ == '__main__':
    plac.call(main)
