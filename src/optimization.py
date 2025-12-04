import optuna
import numpy as np
import pandas as pd
from dotenv import dotenv_values

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from optuna.visualization.matplotlib import plot_param_importances

from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)


def ml_objective(trial, trial_inputs, trial_labels):

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 1, 100, step=1),
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=1),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20, step=1),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.6, 0.8]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 40, step=1),
        "max_samples": trial.suggest_float("max_samples", 0.1, 1, step=0.1)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_predicts = np.empty(5)

    for idx, (train_idx, test_idx) in enumerate(cv.split(trial_inputs, trial_labels)):
        train_inputs, test_inputs = trial_inputs[train_idx], trial_inputs[test_idx]
        train_labels, test_labels = trial_labels[train_idx], trial_labels[test_idx]

        model = RandomForestRegressor(**param_grid,
                                      max_depth = 10,
                                      n_jobs = -1,
                                      random_state = 42)

        model.fit(train_inputs, train_labels)
        preds = model.predict(test_inputs)
        score = root_mean_squared_error(test_labels, preds)
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
        study.optimize(func, n_trials=1000, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted. Saving progress...")

    study_df = study.trials_dataframe()
    study_df.to_csv(env_config["OPTUNA_DATA_PATH"], index=False)

    plot_param_importances(study)
    plt.show()

    print("\n\n" + "-" * 50)
    print(study.best_value)
    print(study.best_params)