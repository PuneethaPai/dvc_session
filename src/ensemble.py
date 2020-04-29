import os

import matplotlib.pyplot as plt
import pandas as pd
import plac
from dagshub import dagshub_logger
from joblib import dump, load
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import plot_confusion_matrix


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    model_path=("Path to save trained Model", "option", "m", str),
    out_path=("Path to save trained Model", "option", "o", str)
)
def main(data_path='data/features/', model_path='data/models/', out_path='data/models/ensemble/'):
    train = pd.read_csv(f'{data_path}train.csv')
    test = pd.read_csv(f'{data_path}test.csv')

    X_train, y_train = train.drop(columns=['class']), train['class']
    X_test, y_test = test.drop(columns=['class']), test['class']

    cl1 = load_model(model_path, 'logistic')
    cl2 = load_model(model_path, 'svc')
    cl3 = load_model(model_path, 'r_forrest')
    estimators = [
        ('l_regression', cl1),
        ('l_svc', cl2),
        ('r_forrest', cl3)
    ]

    model = VotingClassifier(estimators)
    model.fit(X_train, y_train)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    dump(model, f'{out_path}model.pkl')

    cmd = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Reds)
    cmd.figure_.savefig(f'{out_path}confusion_matrix.svg', format='svg')
    c_matrix = cmd.confusion_matrix
    accuracy = model.score(X_test, y_test)

    print(f'Finished Training Ensemble Model:\nStats:')
    print(f'\tConfusion Matrix:\n{c_matrix}')
    print(f'\tModel Accuracy: {accuracy}')
    with dagshub_logger(metrics_path=f'{out_path}theBestMetric.csv',
                        hparams_path=f'{out_path}theBestParams.yaml') as logger:
        logger.log_hyperparams(voting=model.voting)
        logger.log_metrics(accuracy=accuracy)


def load_model(model_path, model_name):
    return load(f'{model_path}{model_name}/model.pkl')


if __name__ == '__main__':
    plac.call(main)
