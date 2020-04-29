import os

import matplotlib.pyplot as plt
import pandas as pd
from dagshub import dagshub_logger
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix


def log_experiment(out_path, params: dict, metrics: dict):
    with dagshub_logger(metrics_path=f'{out_path}metrics.csv', hparams_path=f'{out_path}params.yml') as logger:
        logger.log_hyperparams(params=params)
        logger.log_metrics(metrics=metrics)


def print_results(accuracy, c_matrix, model_name=''):
    print(f'Finished Training {model_name}:\nStats:')
    print(f'\tConfusion Matrix:\n{c_matrix}')
    print(f'\tModel Accuracy: {accuracy}')


def evaluate_model(model, X_test, y_test):
    cmd = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Reds)
    c_matrix = cmd.confusion_matrix
    accuracy = model.score(X_test, y_test)
    return accuracy, c_matrix, cmd.figure_


def save_results(out_path, model, fig):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    dump(model, f'{out_path}model.gz')
    fig.savefig(f'{out_path}confusion_matrix.svg', format='svg')


def read_data(data_path: str) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    train = pd.read_csv(f'{data_path}train.csv')
    test = pd.read_csv(f'{data_path}test.csv')
    X_train, y_train = train.drop(columns=['class']), train['class']
    X_test, y_test = test.drop(columns=['class']), test['class']
    return X_train, X_test, y_train, y_test


def load_model(path):
    return load(f'{path}/model.gz')
