import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import dotenv_values

from scores import get_scores
from config import ENV_FILE_PATH
from preprocessing import preprocess_data
from plotting_utils import plot_learning_curves
from mlflow_utils import run_mlflow_tracking

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

np.random.seed(42)


def train(plot=False, mlflow_tracking=False) -> None:
    """
    Trains a specified model on the titanic dataset
    Saves metrics and model metadata using mlflow tracking in case mlflow_tracking is True
    :param plot: bool value, if True - plots the training metrics
    :param mlflow_tracking: bool value, if True - tracks training experiment
    :return: None
    """

    # Loading data
    env_config = dotenv_values(ENV_FILE_PATH)
    df = pd.read_csv(env_config["TRAIN_DATA_PATH"])

    # Preprocessing data
    train_inputs, train_labels = preprocess_data(df=df, split=True)

    # Model initialization
    xgb = XGBRegressor(max_depth=3,
                       n_estimators=981,
                       max_leaves=36,
                       grow_policy="lossguide",
                       learning_rate=0.021370,
                       gamma=0.011646,
                       min_child_weight=1.203000,
                       subsample=0.30,
                       colsample_bytree=0.65,
                       colsample_bylevel=0.9,
                       colsample_bynode=0.5,
                       reg_alpha=0.066957,
                       reg_lambda=5.423658,
                       objective="reg:squarederror",
                       random_state=42,
                       n_jobs=-1)

    svr = SVR(kernel="rbf", gamma="auto", C=0.519349, epsilon=0.036587)

    model = VotingRegressor(estimators=[('xgb', xgb),
                                        ('svr', svr)])

    # Fetching metrics using cross validation
    train_metrics, val_metrics = get_scores(model, train_inputs, train_labels)

    print("Training" + "\n" + "-" * 30)
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nValidation" + "\n" + "-" * 30)
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")

    # Plotting and saving metrics if plot is True
    if plot:
        save_plot = True if mlflow_tracking else False
        plot_learning_curves(model, train_inputs, train_labels, 5, save_plot)

    # Training model and loading it into the dump
    model.fit(train_inputs, train_labels)
    model_path = Path(env_config["MODEL_DUMP_PATH"])

    if model_path.exists():
        os.remove(model_path)

    joblib.dump(model, env_config["MODEL_DUMP_PATH"])

    # Running mlflow tracking in case mlflow tracking is True
    if mlflow_tracking:
        tags = {
            "Model": "VotingRegressor",
            "DatasetVersion": "V2",
            "PreprocessVersion": "V2"
        }
        run_mlflow_tracking(
            model=model,
            model_name=tags["Model"],
            inputs=train_inputs,
            tracking_uri=env_config["MLFLOW_RUNS_PATH"],
            experiment_name=tags["Model"],
            run_name=f"Run: DatasetVersion {tags['DatasetVersion']}, Preprocessing {tags['PreprocessVersion']}",
            tags=tags,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            plot_path="plot.png"
        )


if __name__ == "__main__":
    train(plot=True, mlflow_tracking=True)