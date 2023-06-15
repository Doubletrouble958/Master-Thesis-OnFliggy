#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
#%%
# Read the data
data = pd.read_excel('D:\\ProgramData\\Lund\\Thesis\\data\\simdata_mini.xls')

# Encode behaviors as integers
le = LabelEncoder()
data['behavior_code'] = le.fit_transform(data['behavior'])

# Filter out sessions that have length less than 3 or more than 250
grouped = data.groupby('index')
data = grouped.filter(lambda x: 3 <= len(x) <= 250)

#identify daytime and night time
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['daytime'] = (data['timestamp'].dt.hour >= 6) & (data['timestamp'].dt.hour < 18)
data['daytime'] = data['daytime'].astype(int)  # convert boolean to integer (1 or 0)

# identify weekend or weekday
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)  # 5 and 6 correspond to Saturday and Sunday

# Compute the time spent in each action using shift()
data['next_timestamp'] = data['timestamp'].shift(-1)
data['next_index'] = data['index'].shift(-1)
time_diff = pd.to_timedelta(data['next_timestamp'] - data['timestamp'])
data['time_spent'] = time_diff.where(data['index'] == data['next_index'], pd.Timedelta(0)).dt.total_seconds()

# Drop the auxiliary columns
data.drop(columns=['next_timestamp', 'next_index'], inplace=True)

# Compute the average time spent for each behavior code
behavior_avg_time = data[data['time_spent'] != 0].groupby('behavior_code')['time_spent'].mean()
# Add the average time spent to the original data
data['average_time'] = data['behavior_code'].map(behavior_avg_time)
data['time_spent'] = data.apply(lambda x: x['average_time'] if x['time_spent'] == 0 else x['time_spent'], axis=1)
#%%
# Group the data by user sessions
session_data = data.groupby('index').agg({'time_spent': lambda x: list(x),
                                          'behavior_code': lambda x: list(x),
                                          'daytime': lambda x: list(x),
                                          'is_weekend': lambda x: list(x)})

input_seq = []
target_seq = []

# Define the prediction window size
for _, row in session_data.iterrows():
    session_sequence = []
    behavior_codes = row['behavior_code']
    time_spent = row['time_spent']
    daytime = row['daytime'] 
    is_weekend = row['is_weekend']
    
    for i, behavior_code in enumerate(behavior_codes):
        # Create a vector with zeros
        vector = [0] * (len(le.classes_) + 2)  # add two additional positions for daytime and is_weekend

       
        vector[behavior_code] = 1 if time_spent[i] > 0 else 0

        # Set the value at the last but one position to daytime (1 for daytime, 0 for nighttime)
        vector[-2] = daytime[i]

        # Set the value at the last position to is_weekend (1 for weekend, 0 for weekday)
        vector[-1] = is_weekend[i]

        session_sequence.append(vector)

    # Remove the last element for the input sequence
    input_seq.append(session_sequence[:-1])

    # Only the last element for the target sequence
    target_seq.append(session_sequence[-1])
#%%
# Convert your sequences to tensors and put them in a list
input_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in input_seq]
target_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in target_seq]

# Pad the sequences
padded_input = pad_sequence(input_tensors, batch_first=True, padding_value=0)

# Convert the target tensors to a tensor
target_tensor = torch.stack(target_tensors)
#%%
# Set your hyperparameters
n_features = len(input_seq[0][0])
hidden_units = 64
batch_size = 32
epochs = 100
learning_rate = 0.001
patience = 10
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

#%%
# Split the data into training (80%), validation (10%), and test (10%) sets
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
X = padded_input
y = target_tensor

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
#%%
# Initialize the model, loss, and optimizer
model = LSTM(n_features, hidden_units, 1, n_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model with early stopping
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        input_batch = X_train[i:i + batch_size]
        target_batch = y_train[i:i + batch_size]

        # Forward pass
        output_batch = model(input_batch)
        loss = criterion(output_batch, target_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print epoch loss
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Validate the model
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        print(f"Validation Loss: {val_loss.item()}")

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
#%%
# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_output = model(X_test)
    test_loss = criterion(test_output, y_test)
    print(f"Test Loss: {test_loss.item()}")

# Plotting confusion matrix
from sklearn.metrics import classification_report

# Convert the predictions to class labels
y_test_np = y_test.cpu().detach().numpy()
test_output_np = test_output.cpu().detach() .numpy()

y_pred = np.argmax(test_output_np, axis=1)
y_true = np.argmax(y_test_np, axis=1)

report = classification_report(y_true, y_pred, zero_division=0)
print(report)

# %%
