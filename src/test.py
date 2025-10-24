import joblib
import numpy as np
import pandas as pd
from config import ENV_FILE_PATH
from dotenv import dotenv_values
from preprocessing import preprocess_data


def kaggle_test() -> None:
    """
    Runs trained model on test data and saved results to submission file
    :return: None
    """

    # Initializing dataframe
    env_config = dotenv_values(ENV_FILE_PATH)
    df_test = pd.read_csv(env_config["TEST_DATA_PATH"])
    ids = df_test["Id"].tolist()

    # Preprocessing dataframe
    test_inputs = preprocess_data(df_test)

    # Loading model from the dump
    model = joblib.load(env_config["MODEL_DUMP_PATH"])

    # Making predictions on test data
    log_y_pred = model.predict(test_inputs)
    y_pred = np.exp(log_y_pred)

    # Saving results to submission file
    submission_data = {"Id": ids, "SalePrice": y_pred}
    df = pd.DataFrame(submission_data)
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    kaggle_test()