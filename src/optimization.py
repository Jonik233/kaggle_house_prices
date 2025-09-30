import optuna
import numpy as np
import pandas as pd
from dotenv import dotenv_values

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_log_error

import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from optuna.visualization.matplotlib import plot_param_importances

from sklearn.linear_model import LinearRegression

np.random.seed(42)


def ml_objective(trial, trial_inputs, trial_labels):

    param_grid = {
        "penalty": trial.suggest_categorical("penalty", ['l1', 'l2']),
        "C": trial.suggest_float("C", 1, 20, step=1),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "solver": trial.suggest_categorical("solver", ['liblinear']),
        "max_iter": trial.suggest_int("max_iter", 10, 1000, step=10)
    }

    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    cv_predicts = np.empty(4)

    for idx, (train_idx, test_idx) in enumerate(cv.split(trial_inputs, trial_labels)):
        train_inputs, test_inputs = trial_inputs[train_idx], trial_inputs[test_idx]
        train_labels, test_labels = trial_labels[train_idx], trial_labels[test_idx]

        model = LinearRegression(**param_grid)

        model.fit(train_inputs, train_labels)
        preds = model.predict(test_inputs)
        score = root_mean_squared_log_error(test_labels, preds)
        cv_predicts[idx] = score

        trial.report(score, idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(cv_predicts)


if __name__ == "__main__":
    env_config = dotenv_values("./.env")
    df = pd.read_csv(env_config["TRAIN_DATA_PATH"])

    inputs, labels = preprocess_data(df=df, split=True)

    func = lambda trial: ml_objective(trial, inputs, labels)
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.HyperbandPruner()
    )

    try:
        study.optimize(func, n_trials=600, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted. Saving progress...")

    study_df = study.trials_dataframe()
    study_df.to_csv(env_config["OPTUNA_DATA_PATH"], index=False)

    plot_param_importances(study)
    plt.show()

    print("\n\n" + "-" * 50)
    print(study.best_value)
    print(study.best_params)