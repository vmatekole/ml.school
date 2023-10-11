
import os
import json
import tarfile
import numpy as np
import pandas as pd
import argparse


from pathlib import Path
# from tensorflow import keras
import torch

from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "/opt/ml/processing/model/"
TEST_PATH = "/opt/ml/processing/test/"
OUTPUT_PATH = "/opt/ml/processing/evaluation/"
EVALUATION_NAME="evaluation"

# Had to repeat class here due to bug in using source_dir param in pytorch_processor.run
class PenguinModel(nn.Module):
    def __init__(self, input_shape):
        super(PenguinModel, self).__init__()
        self.fc1 = nn.Linear(in_features=input_shape, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along dimension 1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    # Prediction function
    def predict(self, input_data):
        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            output = self(input_data)        
        return output
    
def evaluate(model_path, test_path, output_path, evaluation_name):
    # The first step is to extract the model package so we can load 
    # it in memory.
    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))
        
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test.drop(X_test.columns[-1], axis=1, inplace=True)
    
    model = PenguinModel(X_test.shape[1])
    model.load_state_dict(torch.load(Path(model_path) / "001" / "model.pth"))
    input_data = torch.tensor(X_test.values, dtype=torch.float32)

    predictions = np.argmax(model.predict(input_data), axis=-1)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    num_samples = X_test.shape[0]
    
    # print(f"Accuracy: {accuracy}. Precision: {precision}, Recall: {recall}, F1: {f1}, num_samples: {num_samples}")

    # Let's create an evaluation report using the model accuracy.
    evaluation_report = {
        "metrics": {
            "accuracy": {
                "value": accuracy
            },
            "Precision": {
                "value": precision
            },
            "Recall": {
                "value": recall
            },
            "F1": {
                "value": f1
            },
            "num_samples": {
                "value": num_samples
            }
        },
    }
    print(evaluation_report)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / f"{evaluation_name}.json", "w") as f:
        f.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_name', type=str, dest='evaluation_name', default="evaluation")

    args, _ = parser.parse_known_args()
    evaluate(
        model_path=MODEL_PATH, 
        test_path=TEST_PATH,
        output_path=OUTPUT_PATH,
        evaluation_name=args.evaluation_name
    )
