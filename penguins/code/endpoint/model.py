
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Define a custom PyTorch model
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
