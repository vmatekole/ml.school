
import os
import json
import tarfile
import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow import keras
from sklearn.metrics import accuracy_score

from typing import Tuple
# from train import load_csv

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


MODEL_PATH = "/opt/ml/processing/model"
TEST_PATH = "/opt/ml/processing/test"
OUTPUT_PATH = "/opt/ml/processing/evaluation"

def load_csv(file_path,  sample=None) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    num_classes = 10
    
    df = pd.read_csv(file_path)
    df_X_train = df.sample(frac=sample, random_state=1)
    df_X_test = df.sample(frac=sample, random_state=1)
    # X_train = np.genfromtxt(directory / 'train' / 'mnist_train.csv', delimiter=',')
    # X_test = np.genfromtxt(directory / 'test' / 'mnist_test.csv', delimiter=',')
    

    y_train = df_X_train.values[:, -num_classes:]
    y_test = df_X_test.values[:, -num_classes:]
    
    
    return df_X_train.values, y_train, df_X_test.values, y_test

def evaluate(model_path, test_path, output_path):
    # The first step is to extract the model package so we can load 
    # it in memory.
    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))
        
    model = keras.models.load_model(Path(model_path) / "001")
    
    X_train, y_train, X_test, y_test = load_csv(Path(test_path) / "mnist_test.csv", 0.5)

    
    predictions = np.argmax(model.predict(X_test), axis=-1)

#   TODO: Comeback to this 
    encoder = OneHotEncoder()
    predictions = encoder.fit_transform(predictions.reshape(-1,1)) # Reshape to a 2-D array
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy}")

    # Let's create an evaluation report using the model accuracy.
    evaluation_report = {
        "metrics": {
            "accuracy": {
                "value": accuracy
            },
        },
    }
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    evaluate(
        model_path=MODEL_PATH, 
        test_path=TEST_PATH,
        output_path=OUTPUT_PATH
    )
