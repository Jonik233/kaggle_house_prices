import optuna
import numpy as np
import pandas as pd
from dotenv import dotenv_values

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from optuna.visualization.matplotlib import plot_param_importances

from xgboost import XGBRegressor

np.random.seed(42)


def ml_objective(trial, trial_inputs, trial_labels):

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 1, 1000, step=1),
        "max_leaves": trial.suggest_int("max_leaves", 2, 80, step=1),
        "grow_policy": trial.suggest_categorical("grow_policy", ['depthwise', 'lossguide']),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 10, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 1, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1, step=0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5, log=True)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_predicts = np.empty(5)

    for idx, (train_idx, test_idx) in enumerate(cv.split(trial_inputs, trial_labels)):
        train_inputs, test_inputs = trial_inputs[train_idx], trial_inputs[test_idx]
        train_labels, test_labels = trial_labels[train_idx], trial_labels[test_idx]

        model = XGBRegressor(**param_grid, max_depth=5, objective="reg:squarederror", random_state=42, n_jobs=-1)

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