#  Pytorch training script
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

import os
import argparse

from pathlib import Path
import numpy as np
import pandas as pd

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

def train(base_directory, train_path, validation_path, epochs=50, batch_size=32, learning_rate=0.01):
    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train.drop(X_train.columns[-1], axis=1, inplace=True)
    
    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation[X_validation.columns[-1]] # Get the last column of the training dataset
    X_validation.drop(X_validation.columns[-1], axis=1, inplace=True)
   
    # Convert data to PyTorch tensors
    X_train_torch = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train.values, dtype=torch.long)
    X_validation_torch = torch.tensor(X_validation.values, dtype=torch.float32)
    y_validation_torch = torch.tensor(y_validation.values, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # Initialize the model, loss, and optimizer
    model = PenguinModel(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            total_loss += loss 
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
         # Calculate accuracy and average loss for the epoch
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        print(f"Epoch [{epoch+1}/{epochs}] - loss: {epoch_loss:.4f}, val_accuracy: {epoch_accuracy:.4f}")
        
    # Save model
    model_path = Path(base_directory) / 'model' / '001'
    model_path.mkdir(parents=True,exist_ok=True)
    
    torch.save(model.state_dict(), model_path / 'model.pth')
    
    print(f'Model saved: {model_path.resolve()}/model.pth')
   
if __name__ == "__main__":
    # Any hyperparameters provided by the training job are passed to the entry point
    # as script arguments. SageMaker will also provide a list of special parameters
    # that you can capture here. Here is the full list: 
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/params.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default="/opt/ml/")
    # SageMaker will automatically create env variables(prefixed with SM_CHANNEL_) for the training inputs defined in the training step further below
    parser.add_argument("--train_path", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", None))  
    parser.add_argument("--validation_path", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", None))
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    args, _ = parser.parse_known_args()
    
    train(
        base_directory=args.base_directory,
        train_path=args.train_path,
        validation_path=args.validation_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
