import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, KFold


def plot_learning_curves(model, train_data: np.ndarray, train_labels: np.ndarray, cv: int, save_plot: bool = False):

    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)

    train_sizes, train_rmse, val_rmse = learning_curve(
        model, train_data, train_labels,
        shuffle=True, cv=cv_strategy, scoring="neg_root_mean_squared_error"
    )

    plt.plot(train_sizes, np.mean(-train_rmse, axis=1), label="Train RMSE")
    plt.plot(train_sizes, np.mean(-val_rmse, axis=1), label="Validation RMSE")
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    plt.title("Learning Curve - RMSE")
    plt.legend()

    if save_plot:
        plt.savefig("plot.png")

    plt.show()