import plac
from sklearn.ensemble import RandomForestClassifier

from utils import evaluate_model, print_results, save_results, log_experiment, read_data


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    n_estimators=("Path to save trained Model", "option", "e", str),
    max_samples=("Path to save trained Model", "option", "s", str),
    out_path=("Path to save trained Model", "option", "o", str)
)
def main(data_path='data/features/', out_path='data/models/r_forrest/', n_estimators=10, max_samples=30):
    X_train, X_test, y_train, y_test = read_data(data_path)

    name = 'RandomForrest'
    model = RandomForestClassifier(n_estimators=n_estimators, max_samples=max_samples, n_jobs=4)
    model.fit(X_train, y_train)

    accuracy, c_matrix, fig = evaluate_model(model, X_test, y_test)
    print_results(accuracy, c_matrix, name)

    save_results(out_path, model, fig)
    log_experiment(out_path, params=dict(name=name, n_estimators=n_estimators, max_samples=max_samples),
                   metrics=dict(accuracy=accuracy, confusion_matrics=c_matrix))


if __name__ == '__main__':
    plac.call(main)
