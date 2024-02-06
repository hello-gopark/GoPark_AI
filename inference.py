import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import datetime

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

# Step 0 (Not Necessary): Get Example data [here use some part of the dataset to show]
file_path = 'parking_klcc_2016to2017.txt'
df = pd.read_csv(file_path, delimiter=';')
df = df[df['Vacancy'] != 'OPEN']
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['Vacancy'] = df['Vacancy'].apply(lambda x: 0 if x == 'FULL' else int(x))  # Set "FULL" to zero and convert to int

# Step 1: Load and preprocess the new data
new_data = [df['Vacancy'].values[-60:]]  # For query data

# Preprocess the new data using the same scaler used during training
scaler = MinMaxScaler(feature_range=(0, 1))
new_data = scaler.fit_transform(new_data)

print("---Completed preprocessing----\n")

# Step 2: Load GoPark_DNN model
input_size = 1  # Number of features (Vacancy)
hidden_size = 30
num_layers = 3
output_size = 1  # Number of predicted time values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('GoPark_DNN.pth'))
model.eval()

# Step 3: Perform predictions on the new data
new_data_tensor = torch.Tensor(new_data).unsqueeze(dim=2).to(device)
with torch.no_grad():
    predictions = model(new_data_tensor)

# Step 4: Post-process the predictions
predictions = predictions.squeeze().cpu().numpy()

# Print the predictions
time_change = datetime.timedelta(minutes=float(predictions))
print("Now, it is", datetime.datetime.now())
print("Will be full by", datetime.datetime.now()+time_change)