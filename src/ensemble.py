import plac
from sklearn.ensemble import VotingClassifier

from utils import read_data, load_model, evaluate_model, print_results, save_results, log_experiment, read_params


@plac.annotations(
    data_path=("Path to source data", "option", "i", str),
    model_path=("Path to save trained Model", "option", "m", str),
    out_path=("Path to save trained Model", "option", "o", str)
)
def main(data_path='data/features/', model_path='data/models/', out_path='data/models/ensemble/'):
    X_train, X_test, y_train, y_test = read_data(data_path)

    name = 'Ensemble'
    params = read_params('params.yaml', 'ensemble')
    cl1 = load_model(f'{model_path}/logistic/')
    cl2 = load_model(f'{model_path}/svc/')
    cl3 = load_model(f'{model_path}/r_forrest/')
    estimators = [
        ('l_regression', cl1),
        ('l_svc', cl2),
        ('r_forrest', cl3)
    ]

    model = VotingClassifier(estimators, **params)
    model.fit(X_train, y_train)

    accuracy, c_matrix, fig = evaluate_model(model, X_test, y_test)
    print_results(accuracy, c_matrix, name)

    save_results(out_path, model, fig)
    log_experiment(out_path, params=params,
                   metrics=dict(accuracy=accuracy, confusion_matrics=c_matrix))


if __name__ == '__main__':
    plac.call(main)
