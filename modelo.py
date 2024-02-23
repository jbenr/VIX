import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tabulate import tabulate,tabulate_formats

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)  # No need to slice the last element
        return out

def do_the_thing(X, y):
    # Splitting data into train and test sets
    split = int(0.8 * len(X))  # 80% train, 20% test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scaling the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Convert data into sequences with a sequence length of 5
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    seq_length = 10

    X_train_seq = create_sequences(X_train_scaled, seq_length)
    X_test_seq = create_sequences(X_test_scaled, seq_length)

    y_train_seq = create_sequences(y_train_scaled, seq_length)
    y_test_seq = create_sequences(y_test_scaled, seq_length)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train_seq).float()
    X_test_tensor = torch.from_numpy(X_test_seq).float()

    y_train_tensor = torch.from_numpy(y_train_seq).float()
    y_test_tensor = torch.from_numpy(y_test_seq).float()

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model parameters
    input_size = X_train_seq.shape[2]  # Number of features
    hidden_size = 64
    output_size = y_train_seq.shape[2]  # Number of targets
    num_layers = 5

    # Initialize the model
    model = RNN(input_size, hidden_size, output_size, num_layers)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop with loss printing
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.detach().numpy())

    predictions = np.concatenate(predictions, axis=0)
    # Reshape predictions array
    predictions = predictions.reshape(-1, output_size)
    # Inverse scaling
    predictions = scaler_y.inverse_transform(predictions).round(2)
    # print(predictions)
    # Select only the last predictions corresponding to the test data
    last_predictions = predictions[-len(y_test):]

    # Compute evaluation metrics (e.g., RMSE)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test, last_predictions))

    print("Root Mean Squared Error (RMSE):", rmse)

    # Backtesting visualization
    preds = pd.DataFrame(last_predictions, columns=y_test.columns, index=y_test.index)

    print('y_test')
    print(tabulate(y_test.tail(5),headers='keys',tablefmt=tabulate_formats[1]))
    print('preds')
    print(tabulate(preds.tail(5), headers='keys', tablefmt=tabulate_formats[1]))

    import matplotlib.pyplot as plt

    def visualize_predictions(y_test, preds, columns_to_plot, num_plots):
        test = y_test[columns_to_plot]
        pred = preds[columns_to_plot]
        for date in y_test.iloc[-num_plots:].index:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(test.loc[date], label=f'Actual')
            ax.plot(pred.loc[date], label=f'Predicted')
            ax.set_title(f'Predictions vs Actuals for {date}')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()
            plt.show()

    # Assuming you have DataFrame predictions and actuals, and columns_to_plot list
    # columns_to_plot = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7']
    columns_to_plot = ['1-2', '1-3', '1-4', '1-5', '1-6', '1-7']

    visualize_predictions(y_test, preds, columns_to_plot, 10)



