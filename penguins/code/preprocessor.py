
## Preprocessing script
import os
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from pickle import dump


# This is the location where the SageMaker Processing job
# will save the input dataset.
BASE_DIRECTORY = "/opt/ml/processing"
DATA_FILEPATH = Path(BASE_DIRECTORY) / "input" / "data.csv"


def _save_splits(base_directory, train, validation, test):
    """
    One of the goals of this script is to output the three
    dataset splits. This function will save each of these
    splits to disk.
    """

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)


def _save_pipeline(base_directory, pipeline):
    """
    Saves the Scikit-Learn pipeline that we used to
    preprocess the data.
    """
    pipeline_path = Path(base_directory) / "pipeline"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / "pipeline.pkl", "wb"))


def _save_classes(base_directory, classes):
    """
    Saves the list of classes from the dataset.
    """
    path = Path(base_directory) / "classes"
    path.mkdir(parents=True, exist_ok=True)

    np.asarray(classes).tofile(path / "classes.csv", sep=",")


def _save_baseline(base_directory, df_train, df_test):
    """
    During the data and quality monitoring steps, we will need a baseline
    to compute constraints and statistics. This function will save that
    baseline to the disk.
    """

    for split, data in [("train", df_train), ("test", df_test)]:
        baseline_path = Path(base_directory) / f"{split}-baseline"
        baseline_path.mkdir(parents=True, exist_ok=True)

        df = data.copy().dropna()
        df.to_json(
            baseline_path / f"{split}-baseline.json", orient="records", lines=True
        )


def preprocess(base_directory, data_filepath):
    """
    Preprocesses the supplied raw dataset and splits it into a train,
    validation, and a test set.
    """

    df = pd.read_csv(data_filepath)

    numeric_features = df.select_dtypes(include=['float64']).columns.tolist()
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, ["island"]),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor)
        ]
    )

    df.drop(["sex"], axis=1, inplace=True)
    df = df.sample(frac=1, random_state=42)

    df_train, temp = train_test_split(df, test_size=0.3)
    df_validation, df_test = train_test_split(temp, test_size=0.5)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train.species)
    y_validation = label_encoder.transform(df_validation.species)
    y_test = label_encoder.transform(df_test.species)
    
    _save_baseline(base_directory, df_train, df_test)

    df_train = df_train.drop(["species"], axis=1)
    df_validation = df_validation.drop(["species"], axis=1)
    df_test = df_test.drop(["species"], axis=1)

    X_train = pipeline.fit_transform(df_train)
    X_validation = pipeline.transform(df_validation)
    X_test = pipeline.transform(df_test)

    train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
    validation = np.concatenate((X_validation, np.expand_dims(y_validation, axis=1)), axis=1)
    test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)

    _save_splits(base_directory, train, validation, test)
    _save_pipeline(base_directory, pipeline=pipeline)
    _save_classes(base_directory, label_encoder.classes_)


if __name__ == "__main__":
    preprocess(BASE_DIRECTORY, DATA_FILEPATH)
