import json
import numpy as np
import torch
import torch.nn as nn

def build_model(num_features, n_units, num_hidden_layers, num_output, dropout_rate, activation_function, use_batch_norm):
    # Build the ML model
    layers = []

    # Select activation function
    if activation_function == 'ReLU':
        activation = nn.ReLU()
    elif activation_function == 'LeakyReLU':
        activation = nn.LeakyReLU()
    elif activation_function == 'Tanh':
        activation = nn.Tanh()

    # Input layer
    layers.append(nn.Linear(num_features, n_units))
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(n_units))
    layers.append(activation)
    layers.append(nn.Dropout(dropout_rate))

    # Hidden layers
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(n_units, n_units))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(n_units))
        layers.append(activation)
        layers.append(nn.Dropout(dropout_rate))

    # Output layer
    layers.append(nn.Linear(n_units, num_output))
    model = nn.Sequential(*layers)
    return model

def initialize_weights(model, init_method):
    def init_func(m):
        if isinstance(m, nn.Linear):
            if init_method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_method == 'he':
                nn.init.kaiming_uniform_(m.weight)
            elif init_method == 'default':
                nn.init.uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init_func)

def load_best_params(best_params_path):
    # Load the best hyperparameters
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    print("Loaded best hyperparameters: ")
    return best_params

def preprocess_dataset(X, y):
    # Convert data into pytorch tensors
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))

    # Combine features and labels into TensorDataset
    train_dataset = torch.utils.data.TensorDataset(X, y)

    return train_dataset 

def calculate_loss(model, data_loader, device):
    # Calculate the loss of the dataloader
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def predict(model, data_loader, device):
    # Gathe the predictions of the dataloader and return it back
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, outputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())  
    return np.array(predictions)
