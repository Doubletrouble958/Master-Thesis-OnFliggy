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
data = pd.read_csv('D:\\ProgramData\\Lund\\Thesis\\data\\simdata.csv')

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
unique_item_id = data['itemid'].nunique()
unique_category_id = data['categoryid'].nunique()
unique_product_city = data['product city'].nunique()
print("Number of unique item id:", unique_item_id)
print("Number of unique category id:", unique_category_id)
print("Number of unique product city:", unique_product_city)
#%%
import math

item_id_embedding_dim = min(50, round(math.log2(unique_item_id) * 2))
item_category_embedding_dim = min(50, round(math.log2(unique_category_id) * 2))
product_city_embedding_dim = min(50, round(math.log2(unique_product_city) * 2))

# Define the maximum size of each category (the number of unique values in each category)
# You'll need to compute these values from your data
max_item_id =  unique_item_id 
max_item_category = unique_category_id  
max_product_city = unique_product_city  

#%%
# Preprocessing
data['itemid'] = data['itemid'].astype('category').cat.codes
data['categoryid'] = data['categoryid'].astype('category').cat.codes
data['product city'] = data['product city'].astype('category').cat.codes

# Convert the categorical variables to tensors
item_ids = torch.tensor(data['itemid'].values, dtype=torch.long)
category_ids = torch.tensor(data['categoryid'].values, dtype=torch.long)
product_cities = torch.tensor(data['product city'].values, dtype=torch.long)


#%%
class SessionDataset(Dataset):
    def __init__(self, padded_input, item_ids, category_ids, product_cities, target_tensor):
        self.padded_input = padded_input
        self.item_ids = item_ids
        self.category_ids = category_ids
        self.product_cities = product_cities
        self.target_tensor = target_tensor

    def __len__(self):
        return len(self.padded_input)

    def __getitem__(self, idx):
        return self.padded_input[idx], self.item_ids[idx], self.category_ids[idx], self.product_cities[idx], self.target_tensor[idx]
#%%
# Set your hyperparameters
n_features = len(input_seq[0][0])
hidden_units = 64
batch_size = 32
epochs = 100
learning_rate = 0.001
patience = 10
class LSTM(nn.Module):
    def __init__(self, sequence_input_size, item_id_embedding_dim, item_category_embedding_dim, product_city_embedding_dim, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(sequence_input_size + item_id_embedding_dim + item_category_embedding_dim + product_city_embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.item_id_embedding = nn.Embedding(max_item_id, item_id_embedding_dim)
        self.item_category_embedding = nn.Embedding(max_item_category, item_category_embedding_dim)
        self.product_city_embedding = nn.Embedding(max_product_city, product_city_embedding_dim)

    def forward(self, sequences, item_ids, item_categories, product_cities):
        item_id_embedded = self.item_id_embedding(item_ids)
        item_category_embedded = self.item_category_embedding(item_categories)
        product_city_embedded = self.product_city_embedding(product_cities)

        # Expand dimensions of embeddings to match sequence batch size
        item_id_embedded = item_id_embedded.unsqueeze(1).repeat(1, sequences.size(1), 1)
        item_category_embedded = item_category_embedded.unsqueeze(1).repeat(1, sequences.size(1), 1)
        product_city_embedded = product_city_embedded.unsqueeze(1).repeat(1, sequences.size(1), 1)

        x = torch.cat((sequences, item_id_embedded, item_category_embedded, product_city_embedded), -1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#%%
# Split the data into training (80%), validation (10%), and test (10%) sets
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

X = padded_input
y = target_tensor

# Split the categorical variables into training, validation, and test sets
item_id_train, item_id_temp, category_id_train, category_id_temp, product_city_train, product_city_temp = train_test_split(item_ids, category_ids, product_cities, test_size=1 - train_ratio, random_state=42)
item_id_val, item_id_test, category_id_val, category_id_test, product_city_val, product_city_test = train_test_split(item_id_temp, category_id_temp, product_city_temp, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
train_dataset = SessionDataset(X_train, item_id_train, category_id_train, product_city_train, y_train)
val_dataset = SessionDataset(X_val, item_id_val, category_id_val, product_city_val, y_val)
test_dataset = SessionDataset(X_test, item_id_test, category_id_test, product_city_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%%
# Initialize the model, loss, and optimizer
model = LSTM(n_features, item_id_embedding_dim, item_category_embedding_dim, product_city_embedding_dim, hidden_units, 1, n_features)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model with early stopping
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    for sequences, item_ids, item_categories, product_cities, targets in train_loader:
        # Forward pass
        outputs = model(sequences, item_ids, item_categories, product_cities)
        output_sigmoid = torch.sigmoid(outputs)
        loss = criterion(output_sigmoid, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Validate the model
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for sequences, item_ids, item_categories, product_cities, targets in val_loader:
            val_output = model(sequences, item_ids, item_categories, product_cities)
            val_output = torch.sigmoid(val_output)
            val_loss = criterion(val_output, targets)
            total_val_loss += val_loss.item()
    print(f"Validation Loss: {val_loss.item()}")

    avg_val_loss = total_val_loss / len(val_loader)

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

#%%
model.eval()
total_test_loss = 0
test_outputs = []
y_tests = []
with torch.no_grad():
    for sequences, item_ids, item_categories, product_cities, targets in test_loader:
        test_output = model(sequences, item_ids, item_categories, product_cities)
        test_output = torch.sigmoid(test_output)
        test_loss = criterion(test_output, targets)
        total_test_loss += test_loss.item()
        test_outputs.append(test_output)
        y_tests.append(targets)

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss}")

# Concatenate the predictions and targets for all the batches
y_test_np = np.concatenate([y.cpu().detach().numpy() for y in y_tests])
test_output_np = np.concatenate([output.cpu().detach().numpy() for output in test_outputs])


y_pred = np.argmax(test_output_np[:, [0, 1, 2, 3]], axis=1)
y_true = np.argmax(y_test_np[:, [0, 1, 2, 3]], axis=1)

report = classification_report(y_true, y_pred, zero_division=0)
print(report)

#%%
test_output_np
y_pred
y_true
test_outputs
y_test
