import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple
from dotenv import dotenv_values

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    TARGET,
    LOG_SCALE_COLS,
    STANDARD_SCALE_COLS,
    ENV_FILE_PATH,
    COLS_TO_USE,
    COLS_TO_ENCODE,
    NUMERICAL_COLS_TO_FILL,
    CATEGORICAL_COLS_TO_FILL
)

np.random.seed(42)


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills nan values in columns that are specified in CATEGORICAL_COLS_TO_FILL, NUMERICAL_COLS_TO_FILL
    Simple imputer with 'mean' strategy is used for numeric columns
    Simple imputer with 'most_frequent' strategy is used for categorical columns
    :param df: input pandas.DataFrame
    :return: pandas.DataFrame
    """

    # Loading env config and imputers paths
    # env_config = dotenv_values(ENV_FILE_PATH)
    # num_imputer_path = Path(env_config["NUMERIC_IMPUTER_DUMP_PATH"])
    # cat_imputer_path = Path(env_config["CATEGORICAL_IMPUTER_DUMP_PATH"])
    #
    # if num_imputer_path.exists():
    #     # Loading numerical imputer
    #     num_imputer = joblib.load(num_imputer_path)
    # else:
    #     print(f"\nNumerical imputer not found in: {num_imputer_path}")
    #     print("Creating new numerical imputer...")
    #
    #     # Creating new numerical imputer
    #     num_imputer = SimpleImputer(strategy="mean")
    #     num_imputer.fit(df[NUMERICAL_COLS_TO_FILL])
    #
    #     # Loading new numerical imputer into the dump
    #     print("Loading new numerical imputer into the dump...")
    #     joblib.dump(num_imputer, num_imputer_path)
    #
    # if cat_imputer_path.exists():
    #     # Loading categorical imputer
    #     cat_imputer = joblib.load(cat_imputer_path)
    # else:
    #     print(f"\nCategorical imputer not found in: {cat_imputer_path}")
    #     print("Creating new categorical imputer...")
    #
    #     # Creating new categorical imputer
    #     cat_imputer = SimpleImputer(strategy="most_frequent")
    #     cat_imputer.fit(df[CATEGORICAL_COLS_TO_FILL])
    #
    #     # Loading new categorical imputer into the dump
    #     print("Loading new categorical imputer into the dump...")
    #     joblib.dump(cat_imputer, cat_imputer_path)
    #
    # # Applying imputers
    # df[NUMERICAL_COLS_TO_FILL] = num_imputer.transform(df[NUMERICAL_COLS_TO_FILL])
    # df[CATEGORICAL_COLS_TO_FILL] = cat_imputer.transform(df[CATEGORICAL_COLS_TO_FILL])
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies binary encoding to Sex column and OneHotEncoding to columns specified in COLS_TO_ENCODE
    OneHotEncoder is set up with categories specified to be encoded for each particular column
    :param df: input pandas.DataFrame
    :return: pandas.DataFrame
    """

    # Encoding categorical column 'Sex'
    # df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    #
    # # Load config and encoder path
    # env_config = dotenv_values(ENV_FILE_PATH)
    # encoder_path = Path(env_config["UNIVERSAL_ENCODER_DUMP_PATH"])
    #
    # if encoder_path.exists():
    #     # Loading universal encoder
    #     universal_encoder = joblib.load(encoder_path)
    # else:
    #     print(f"\nUniversal encoder not found in {encoder_path}")
    #     print("Creating new encoder...")
    #
    #     # Creating new encoder
    #     universal_encoder = OneHotEncoder(
    #         categories=[None],
    #         handle_unknown="ignore",
    #         sparse_output=False,
    #         dtype=np.int64,
    #     )
    #     universal_encoder.fit(df[COLS_TO_ENCODE])
    #
    #     # Loading new encoder into the dump
    #     print(f"Saving to encoder to {encoder_path}")
    #     joblib.dump(universal_encoder, encoder_path)
    #
    # # Applying universal encoder
    # encoded_arr = universal_encoder.transform(df[COLS_TO_ENCODE])
    # encoded_df = pd.DataFrame(
    #     data=encoded_arr,
    #     columns=universal_encoder.get_feature_names_out(COLS_TO_ENCODE),
    #     index=df.index,
    #     dtype=np.int64,
    # )
    #
    # # Replacing unencoded columns with encoded
    # df = df.drop(columns=COLS_TO_ENCODE)
    # df = pd.concat([df, encoded_df], axis=1)
    return df


def scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standard scaling for STANDARD_SCALE_COLS and log scaling for LOG_SCALE_COLS
    :param df: input pandas.DataFrame
    :return: pandas.DataFrame
    """

    # Loading env config and scaler path
    env_config = dotenv_values(ENV_FILE_PATH)
    scaler_path = Path(env_config["SCALER_DUMP_PATH"])

    if scaler_path.exists():
        # Loading scaler
        scaler = joblib.load(scaler_path)
    else:
        print(f"\nScaler not found in {scaler_path}")
        print("Creating new scaler...")

        # Creating new scaler
        scaler = StandardScaler()
        scaler.fit(df[STANDARD_SCALE_COLS])

        # Loading new scaler into the dump
        print("Loading scaler into the dump...")
        joblib.dump(scaler, scaler_path)

    # Applying scaling
    df[TARGET] = np.log(df[TARGET] + 1) / np.log(100)
    df[LOG_SCALE_COLS] = np.log(df[LOG_SCALE_COLS] + 1) / np.log(100)
    df[STANDARD_SCALE_COLS] = scaler.transform(df[STANDARD_SCALE_COLS])
    return df


def drop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns from input pandas.DataFrame that are not specified in COLS_TO_USE
    :param df: input pandas.DataFrame
    :return: pandas.DataFrame
    """
    data = df[COLS_TO_USE]
    return data


def create_new_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new columns in given pandas.DataFrame
    :param df: input pandas.DataFrame
    :return: pandas.DataFrame
    """
    return df


def preprocess_data(
    df: pd.DataFrame, split: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Applies a pipeline of transformations to a given pandas.DataFrame
    :param df: input pandas.DataFrame
    :param split: bool value, if True - splits on inputs and labels
    :return: np.ndarray
    """

    # Filling nan values using numerical and categorical imputers
    df = fill_na(df)

    # Creating additional data columns
    df = create_new_cols(df)

    # Applying binary encoding and OneHotEncoder(to columns with diverse categories)
    df = encode(df)

    # Applying StandardScaler
    df = scale(df)

    # Dropping redundant columns
    df = drop(df)

    # Splitting DataFrame into numpy arrays of inputs and labels if split is True
    if split:
        labels = df[[TARGET]].values.ravel().astype(np.float32)
        inputs = df.drop(columns=[TARGET]).values.astype(np.float32)
        return inputs, labels

    inputs = df.values.astype(np.float32)
    return inputs