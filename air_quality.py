!pip install streamlit pandas numpy torch matplotlib scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Load the dataset
gatrain_df = pd.read_csv("dataset.csv")
gatrain_df = gatrain_df.drop(columns=['From Date', 'To Date'])  # Adjust column names if needed
gatrain_df = gatrain_df.apply(pd.to_numeric, errors='coerce')
gatrain_df = gatrain_df.dropna()

# Define individual models (CNN, GRU, LSTM)
class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(CNNModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.cnn(x))
        x = x.mean(dim=2)  # Global Average Pooling
        x = self.fc(x)
        return x

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


# Preprocess data function
def preprocess_data(df, target_pollutant, sequence_length=5):
    X = df.drop(columns=[target_pollutant])
    y = df[target_pollutant]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_normalized = scaler_X.fit_transform(X)
    y_normalized = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_sequences = []
    y_sequences = []
    for i in range(len(X_normalized) - sequence_length):
        X_sequences.append(X_normalized[i:i + sequence_length])
        y_sequences.append(y_normalized[i + sequence_length])
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences).reshape(-1)

    X_train, X_val, y_train, y_val = train_test_split(X_sequences, y_sequences, test_size=0.01, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, scaler_X, scaler_y

# Function to train a single model
def train_model(model, train_loader, optimizer, criterion, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()


# Function for genetic ensemble optimization
def optimize_weights(models, train_loader, val_loader):
    def ensemble_loss(weights):
        ensemble_predictions = []
        all_true_values = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                batch_predictions = np.array([model(X_batch).squeeze().detach().numpy() for model in models])
                weighted_predictions = np.tensordot(weights, batch_predictions, axes=([0],[0]))
                ensemble_predictions.extend(weighted_predictions)
                all_true_values.extend(y_batch.numpy())
        return mean_squared_error(all_true_values, ensemble_predictions)

    boundary = [(0, 1)] * len(models)
    result = differential_evolution(ensemble_loss, boundary, maxiter=100, tol=1e-6)
    return result.x

# Function to train ensemble models and compute metrics
def train_ensemble_and_get_metrics(train_loader, val_loader):
    # Initialize models
    cnn_model = CNNModel(input_dim=train_loader.dataset.tensors[0].shape[2], hidden_dim=50)
    gru_model = GRUModel(input_dim=train_loader.dataset.tensors[0].shape[2], hidden_dim=50)
    lstm_model = LSTMModel(input_dim=train_loader.dataset.tensors[0].shape[2], hidden_dim=50)

    models = [cnn_model, gru_model, lstm_model]
    criterion = nn.MSELoss()
    optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

    # Train each model
    for model, optimizer in zip(models, optimizers):
        train_model(model, train_loader, optimizer, criterion)

    # Optimize ensemble weights using genetic algorithm
    optimal_weights = optimize_weights(models, train_loader, val_loader)

    # Ensemble prediction and evaluation using optimized weights
    ensemble_predictions = []
    all_true_values = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            batch_predictions = np.array([model(X_batch).squeeze().numpy() for model in models])
            weighted_predictions = np.tensordot(optimal_weights, batch_predictions, axes=([0],[0]))
            ensemble_predictions.extend(weighted_predictions)
            all_true_values.extend(y_batch.numpy())

    plt.figure(figsize=(10, 6))
    plt.plot(all_true_values, label='Actual Values', color='blue')
    plt.plot( ensemble_predictions, label='Predicted Values', color='red')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    return compute_metrics(all_true_values, ensemble_predictions)


# Metrics computation function
def compute_metrics(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    return mse, mae, rmse, r2

# Define the target pollutant for prediction and preprocess the data
target_pollutant = "PM2.5"
train_loader, val_loader, scaler_X, scaler_y = preprocess_data(gatrain_df, target_pollutant)


def train_ensemble_of_models(train_loader, val_loader):
    # Initialize models
    cnn_model = CNNModel(input_dim=train_loader.dataset.tensors[0].shape[2], hidden_dim=50)
    gru_model = GRUModel(input_dim=train_loader.dataset.tensors[0].shape[2], hidden_dim=50)
    lstm_model = LSTMModel(input_dim=train_loader.dataset.tensors[0].shape[2], hidden_dim=50)

    models = [cnn_model, gru_model, lstm_model]
    criterion = nn.MSELoss()
    optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

    # Train each model
    for model, optimizer in zip(models, optimizers):
        train_model(model, train_loader, optimizer, criterion)

    return cnn_model, gru_model, lstm_model

st.title('Air Quality Prediction')

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file)
    df_processed, scaler = preprocess_data(df, 'PM2.5') # Assuming 'PM2.5' is the target

    # Train models and get predictions
    cnn_model, gru_model, lstm_model = train_ensemble_of_models(df_processed)
    ensemble_results = train_ensemble_and_get_metrics(df_processed)

    # Show results
    st.write(pd.DataFrame({
        'Ensemble': ensemble_results
    }, index=['MSE', 'MAE', 'RMSE', 'R2']))

    # Plotting
    fig, ax = plt.subplots()
    # Add your plotting logic here
    st.pyplot(fig)
