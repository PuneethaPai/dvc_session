import plac
from sklearn.linear_model import LogisticRegression

from utils import evaluate_model, print_results, save_results, log_experiment, read_data


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    out_path=("Path to save trained Model", "option", "o", str)
)
def main(data_path='data/features/', out_path='data/models/logistic/'):
    X_test, X_train, y_test, y_train = read_data(data_path)

    name = 'LogisticRegression'
    model = LogisticRegression(penalty='l2', n_jobs=4)
    model.fit(X_train, y_train)

    accuracy, c_matrix, fig = evaluate_model(model, X_test, y_test)
    print_results(accuracy, c_matrix, name)

    save_results(out_path, model, fig)
    log_experiment(out_path, params=dict(name=name, penalty='l2'),
                   metrics=dict(accuracy=accuracy, confusion_matrics=c_matrix))


if __name__ == '__main__':
    plac.call(main)
