import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, KFold


def plot_learning_curves(model, train_data: np.ndarray, train_labels: np.ndarray, cv: int, save_plot: bool = False):

    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%.*"
        )

        train_sizes, train_rmse, val_rmse = learning_curve(
            model, train_data, train_labels,
            shuffle=True, cv=cv_strategy, scoring="neg_root_mean_squared_error"
        )

        train_sizes, train_rmsle, val_rmsle = learning_curve(
            model, train_data, train_labels,
            shuffle=True, cv=cv_strategy, scoring="neg_root_mean_squared_log_error"
        )

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_sizes, np.mean(-train_rmse, axis=1), label="Train RMSE")
    axes[0].plot(train_sizes, np.mean(-val_rmse, axis=1), label="Validation RMSE")
    axes[0].set_xlabel("Training Samples")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Learning Curve - RMSE")
    axes[0].legend()

    axes[1].plot(train_sizes, np.mean(-train_rmsle, axis=1), label="Train RMSLE")
    axes[1].plot(train_sizes, np.mean(-val_rmsle, axis=1), label="Validation RMSLE")
    axes[1].set_xlabel("Training Samples")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Learning Curve - RMSLE")
    axes[1].legend()

    plt.tight_layout()

    if save_plot:
        plt.savefig("plot.png")

    plt.show()