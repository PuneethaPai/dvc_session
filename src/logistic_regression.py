import os

import pandas as pd
import plac
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import _pickle as cpickle


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    out_path=("Path to save trained Model", "option", "o", str)
)
def main(data_path='data/features/', out_path='data/models/logistic/'):
    train = pd.read_csv(f'{data_path}train.csv')
    test = pd.read_csv(f'{data_path}test.csv')

    X_train, y_train = train.drop(columns=['class']), train['class']
    X_test, y_test = test.drop(columns=['class']), test['class']

    model = LogisticRegression(n_jobs=4)
    model.fit(X_train, y_train)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open(f'{out_path}model.pkl', 'wb+') as fp:
        cpickle.dump(model, fp)

    plot = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Reds).figure_

    plot.savefig(f'{out_path}confusion_matrix.svg', format='svg')
    print(f'Finished Training LogisticRegressionModel:\nStats:')
    print(f'\tConfusion Matrix:\n{confusion_matrix(y_test, model.predict(X_test))}')
    print(f'\tModel Accuracy: {model.score(X_test, y_test)}')


if __name__ == '__main__':
    plac.call(main)
