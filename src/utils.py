import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from dagshub import dagshub_logger
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix
from yaml import CLoader as Loader


def log_experiment(out_path, metrics: dict):
    with dagshub_logger(
        metrics_path=f"{out_path}metrics.csv", should_log_hparams=False
    ) as logger:
        logger.log_metrics(metrics=metrics)


def print_results(accuracy, c_matrix, model_name=""):
    print(f"Finished Training {model_name}:\nStats:")
    print(f"\tConfusion Matrix:\n{c_matrix}")
    print(f"\tModel Accuracy: {accuracy}")


def evaluate_model(model, X_test, y_test):
    cmd = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Reds)
    c_matrix = cmd.confusion_matrix
    accuracy = model.score(X_test, y_test)
    return accuracy, c_matrix, cmd.figure_


def save_results(out_path, model, fig):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    dump(model, f"{out_path}model.gz")
    if fig:
        fig.savefig(f"{out_path}confusion_matrix.svg", format="svg")


def read_data(data_path: str) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    train = pd.read_csv(f"{data_path}train.csv")
    test = pd.read_csv(f"{data_path}test.csv")
    X_train, y_train = train.drop(columns=["class"]), train["class"]
    X_test, y_test = test.drop(columns=["class"]), test["class"]
    return X_train, X_test, y_train, y_test


def load_model(path):
    return load(f"{path}/model.gz")


def read_params(file="params.yaml", model="pca"):
    with open(file, "r") as fp:
        params = yaml.load(fp, Loader)
    return params[model]
