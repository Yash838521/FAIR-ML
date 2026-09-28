"""
preprocessing.py
----------------
Clean, leakage-free preprocessing pipeline for Adult Income dataset.

Fixes:
- No string leakage into model
- Proper encoding (OrdinalEncoder)
- No data leakage in scaling
- Clean train/test split workflow
"""

import os
import urllib.request
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

SENSITIVE_FEATURES = ["gender", "race"]

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "native_country"
]

NUMERICAL_COLS = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]

TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


# ─────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────

def download_data(data_dir: str = None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "../data")

    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "adult.data")
    test_path  = os.path.join(data_dir, "adult.test")

    if not os.path.exists(train_path):
        print(" Downloading adult.data ...")
        urllib.request.urlretrieve(TRAIN_URL, train_path)

    if not os.path.exists(test_path):
        print(" Downloading adult.test ...")
        urllib.request.urlretrieve(TEST_URL, test_path)

    return train_path, test_path


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_data(train_path, test_path=None):
    df = pd.read_csv(
        train_path,
        names=COLUMN_NAMES,
        sep=",",
        skipinitialspace=True,
        na_values="?"
    )

    if test_path:
        df_test = pd.read_csv(
            test_path,
            names=COLUMN_NAMES,
            sep=",",
            skipinitialspace=True,
            na_values="?",
            skiprows=1
        )
        df = pd.concat([df, df_test], ignore_index=True)

    return df


# ─────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────

def clean_data(df):
    df = df.dropna().copy()
    df["income"] = df["income"].str.replace(".", "", regex=False).str.strip()
    return df


# ─────────────────────────────────────────────────────────────
# ENCODING
# ─────────────────────────────────────────────────────────────

def encode_target(df):
    df = df.copy()
    df["income"] = (df["income"] == ">50K").astype(int)
    return df


def encode_sensitive(df):
    df = df.copy()
    df["gender_raw"] = df["gender"]
    df["race_raw"] = df["race"]

    df["gender"] = (df["gender"] == "Male").astype(int)
    df["race_binary"] = (df["race"] == "White").astype(int)

    return df


def encode_categoricals(train_df, test_df):
    """
    Fit encoder ONLY on train (prevents leakage)
    """
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    train_df[CATEGORICAL_COLS] = encoder.fit_transform(train_df[CATEGORICAL_COLS])
    test_df[CATEGORICAL_COLS] = encoder.transform(test_df[CATEGORICAL_COLS])

    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────────────────────────

def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS]  = scaler.transform(X_test[NUMERICAL_COLS])

    return X_train, X_test, scaler


# ─────────────────────────────────────────────────────────────
# FINAL FEATURE BUILDER
# ─────────────────────────────────────────────────────────────

def build_features(df):
    drop_cols = [
        "income",
        "gender_raw",
        "race_raw",
        "race"  
    ]

    X = df.drop(columns=drop_cols).copy()
    y = df["income"].copy()

    sensitive = df[["gender_raw", "race_raw", "gender", "race_binary"]].copy()

    # force numeric safety (no strings allowed)
    X = X.apply(pd.to_numeric)

    return X, y, sensitive

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_preprocessing(train_path, test_path=None, test_size=0.2, random_state=42,
                      validation_size=0.0):

    df = load_data(train_path, test_path)
    df = clean_data(df)
    df = encode_target(df)
    df = encode_sensitive(df)

    # split BEFORE encoding/scaling (BEST PRACTICE)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["income"]
    )

    validation_df = None
    if validation_size:
        train_df, validation_df = train_test_split(
            train_df, test_size=validation_size, random_state=random_state,
            stratify=train_df["income"]
        )
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[CATEGORICAL_COLS] = encoder.fit_transform(train_df[CATEGORICAL_COLS])
    test_df[CATEGORICAL_COLS] = encoder.transform(test_df[CATEGORICAL_COLS])
    if validation_df is not None:
        validation_df[CATEGORICAL_COLS] = encoder.transform(validation_df[CATEGORICAL_COLS])

    # build features
    X_train, y_train, sens_train = build_features(train_df)
    X_test, y_test, sens_test = build_features(test_df)

    # scale numericals safely
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # sanity check (VERY IMPORTANT)
    assert X_train.select_dtypes(include=["object"]).empty, "❌ Still has string columns!"

    feature_names = X_train.columns.tolist()

    result = (
        X_train, X_test,
        y_train, y_test,
        sens_train, sens_test,
        feature_names,
        scaler
    )
    if validation_df is not None:
        X_val, y_val, sens_val = build_features(validation_df)
        X_val[NUMERICAL_COLS] = scaler.transform(X_val[NUMERICAL_COLS])
        return result + (X_val, y_val, sens_val)
    return result


# ─────────────────────────────────────────────────────────────
# TEST RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_path, test_path = download_data()

    X_train, X_test, y_train, y_test, sens_train, sens_test, feats, scaler = \
        run_preprocessing(train_path, test_path)

    print("\n Train size:", len(X_train))
    print(" Test size :", len(X_test))
    print(" Features  :", len(feats))
    print(" Positive rate:", y_train.mean())
    print("\n Preprocessing COMPLETE (no leakage, no strings)")
