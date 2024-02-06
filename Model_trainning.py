import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np

# Step 1: Load and preprocess the dataset
file_path = 'parking_klcc_2016to2017.txt'
df = pd.read_csv(file_path, delimiter=';')
df = df[df['Vacancy'] != 'OPEN']
df = df.drop('KLCC', axis=1)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['Vacancy'] = df['Vacancy'].apply(lambda x: 0 if x == 'FULL' else int(x))  # Set "FULL" to zero and convert to int

# Step 2: Create sequences and labels
sequences = []
labels = []
timestamps = []
count = 0
for i in range(60, len(df)):
    sequences.append(df.iloc[i-60:i]['Vacancy'].values)
    labels.append(df.iloc[i]['Vacancy'])
    if df.iloc[i]['Vacancy'] !=0:
      count+=1
    else:
      for j in range(count+1):
        timestamps.append(df.iloc[i]['timestamp'])
      count=0

sequences = sequences[:len(timestamps)]
timestamps = timestamps[:len(sequences)]

# Step 3: Prepare the data for the LSTM model
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

scaler = MinMaxScaler(feature_range=(0, 1))
sequences = scaler.fit_transform(sequences)

# Convert timestamps to numeric representation (e.g., minutes since the first timestamp)

starting = df['timestamp'].min()
prev = timestamps[0]
for i in range(len(timestamps)):
  if timestamps[i]==prev:
    timestamps[i] = float((timestamps[i]-starting).seconds/60)
  else:
    starting = prev
    prev = timestamps[i]
    timestamps[i] = float((timestamps[i]-starting).seconds/60)

# Step 4: Split the data into training and testing sets
train_size = int(0.8 * len(sequences))
train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]
train_timestamps, test_timestamps = timestamps[:train_size], timestamps[train_size:]

# Step 5: Define and train the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 1  # Number of features (Vacancy)
hidden_size = 30
num_layers = 3
output_size = 1  # Number of predicted time values
num_epochs = 94
batch_size = 32
learning_rate = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_sequences), torch.Tensor(train_timestamps))
train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_sequences), torch.Tensor(test_timestamps))
test_loader= DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for sequences, timestamps in train_loader:
        sequences = sequences.unsqueeze(dim=2).to(device)
        timestamps = timestamps.to(device)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), timestamps)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 6: Evaluate the model
model.eval()
import numpy as np

# Step 6: Evaluate the model
model.eval()
with torch.no_grad():
    loss = 0
    counter = 0
    for sequences, timestamps in test_loader:
        sequences = sequences.unsqueeze(dim=2).to(device)
        timestamps = timestamps.to(device)

        # Forward pass
        test_outputs = model(sequences)
        test_loss = criterion(test_outputs.squeeze(), timestamps.squeeze())
        loss += test_loss.item()
        counter += 1
    print(f'Test Loss: {(loss/count):.4f} equivalent to {np.sqrt(loss/count):.4f}')

torch.save(model.state_dict(), 'GoPark_DNN.pth')