# %%
import importlib
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import general_utils
importlib.reload(general_utils)
from general_utils import *

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Define important variables
n_epochs_optuna = 200
n_epochs_final = 1000
n_trials = 100
epoch_early_stopping = 50

# Define the model path
model_path       = "./inputdecks/model.pth"
best_params_path = "./inputdecks/best_10_hyperparams.json"

# Define the device (mps for macOS)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the database
dataset = pd.read_csv("./inputdecks/ml_database.csv", index_col=0)

# Define the features (input) and targets (output) of the ML model
num_features = 2 # First two columns (E_modul and Load)
num_targets  = 1 # Displacement in the y direction in node 5

# Extract the feature columns 
dataset_features = dataset.iloc[:, 0:num_features].to_numpy()

# Extract the target columns
dataset_target = dataset["Node_5"].to_numpy()[:, np.newaxis]

# Split data into training and validation sets, stratified by "E_modul"
# Stratification ensures that both values of E-Modul (210000 and 70000) are proportionally represented in both splits
X_train, X_dev, y_train, y_dev = train_test_split(
    dataset_features, dataset_target, test_size=0.20, random_state=seed, stratify=dataset["E_modul"]
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev  = scaler.transform(X_dev)

# Preprocess the dataset
train_dataset = preprocess_dataset(X_train, y_train)
val_dataset   = preprocess_dataset(X_dev, y_dev)

def objective(trial, n_epochs_optuna):

    # Hyperparameters to tune for ANN
    n_units = trial.suggest_int("n_units", 32, 128)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    activation_function = trial.suggest_categorical("activation_function", ["ReLU", "LeakyReLU", "Tanh"])
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    init_method = trial.suggest_categorical("init_method", ["default", "xavier", "he"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Build the model and move it to the device, then initialize the weights
    model = build_model(num_features, n_units, num_hidden_layers, num_targets, dropout_rate, activation_function, use_batch_norm).to(device)
    initialize_weights(model, init_method)

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Training
    for epoch in range(n_epochs_optuna):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Calculate validation loss
        val_loss = calculate_loss(model, val_loader, device)

        # Early stopping and pruning 
        if epoch > epoch_early_stopping and trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

# Store best hyperparameters of an optuna study
if Path(best_params_path).exists():

    best_params = load_best_params(best_params_path)

# Begin an optuna study
else:

    # Set up the study and perform the optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, n_epochs_optuna), n_trials=n_trials)

    # Best parameters
    best_params = study.best_params
    print("Best parameters: ", best_params)

    # Save the hyperparameters to a JSON file
    with open(best_params_path, "w") as f:
        json.dump(best_params, f)

# Define the hyperparameters in variables
n_units = best_params["n_units"]
num_hidden_layers = best_params["num_hidden_layers"]
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]
dropout_rate = best_params["dropout_rate"]
optimizer_name = best_params["optimizer"]
activation_function = best_params["activation_function"]
use_batch_norm = best_params["use_batch_norm"]
init_method = best_params["init_method"]
weight_decay = best_params["weight_decay"]

# Build the model
model = build_model(num_features, n_units, num_hidden_layers, num_targets, dropout_rate, activation_function, use_batch_norm).to(device)
initialize_weights(model, init_method)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
if optimizer_name == "SGD":
    momentum = best_params["momentum"]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Check if model exists
if Path(model_path).exists():

    # Load the model
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

# If no model is found, train one 
else:

    # Training
    for epoch in range(n_epochs_final):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Calculate validation loss
        val_loss = calculate_loss(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{n_epochs_final}, Validation Loss: {val_loss}")

    # Save the final model
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved.")

###### POST-PROCESSING ######

# Check how good the model performs in general (for all the dataset (train + validation dataset))

# Create a DataLoader for the full dataset
full_dataset = preprocess_dataset(scaler.transform(dataset_features), dataset_target)  
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# Predict on the full dataset
predictions = predict(model, full_loader, device)

# Add predictions to the original dataset as a new column
dataset["Model_Prediction"] = predictions

# Calculate the absolute percentage error between the actual and predicted displacements for Node_5
# Handle division by zero for cases where Node_5 is zero
dataset["Error_Percentage"] = np.where(
    dataset["Node_5"] != 0,
    (np.abs(dataset["Node_5"] - dataset["Model_Prediction"])) / np.abs(dataset["Node_5"]) * 100,
    0  # Set error to 0 when Node_5 is zero to avoid division by zero
)

# Print summary statistics for the Error_Percentage column
print(f"dataset['Error_Percentage'] = \n{dataset['Error_Percentage'].describe()}")