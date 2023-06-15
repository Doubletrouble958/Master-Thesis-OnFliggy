#%%
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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

#%%
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

# Group the data by user sessions
session_data = data.groupby('index').agg({'time_spent': lambda x: list(x),
                                          'behavior_code': lambda x: list(x),
                                          'daytime': lambda x: list(x),
                                          'is_weekend': lambda x: list(x)})

print(session_data.head(10))


#%%
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

        # Set the value at the index corresponding to the behavior code to the time spent
        vector[behavior_code] = 1 if time_spent[i] > 0 else 0

        # Set the value at the last but one position to daytime (1 for daytime, 0 for nighttime)
        vector[-2] = daytime[i]

        # Set the value at the last position to is_weekend (1 for weekend, 0 for weekday)
        vector[-1] = is_weekend[i]

        session_sequence.append(vector)

    # Remove the last 'prediction_window' elements for the input sequence
    input_seq.append(session_sequence[:-1])

    # Remove the first 'prediction_window' elements for the target sequence
    # Only the last action is kept for the target sequence
    target_seq.append(session_sequence[-1])

    
print(input_seq[:5])
print(target_seq[:5])
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
grouped = data.groupby('index').agg(list)
#%%
from torch.nn.utils.rnn import pad_sequence

# Convert the lists to tensors and pad them
item_ids = pad_sequence([torch.tensor(i, dtype=torch.long) for i in grouped['itemid'].values], batch_first=True)[:,:-1]
category_ids = pad_sequence([torch.tensor(i, dtype=torch.long) for i in grouped['categoryid'].values], batch_first=True)[:,:-1]
product_cities = pad_sequence([torch.tensor(i, dtype=torch.long) for i in grouped['product city'].values], batch_first=True)[:,:-1]

#%%
# Convert your sequences to tensors and put them in a list
input_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in input_seq]
target_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in target_seq]

# Pad the sequences
padded_input = pad_sequence(input_tensors, batch_first=True, padding_value=0)

# Convert the target tensors to a tensor
target_tensor = torch.stack(target_tensors)
# Convert the target sequences to tensors directly
target_seq_tensor = torch.tensor(target_seq, dtype=torch.float32)
target_indices = torch.argmax(target_tensor, dim=1)

#%%
import math
    
class TransformerModelConfig:
    def __init__(self, feature_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        self.feature_size = feature_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

config = TransformerModelConfig(
    feature_size=len(le.classes_) + 2 + item_id_embedding_dim + item_category_embedding_dim + product_city_embedding_dim,  # number of unique behaviors + 2 for daytime and is_weekend
    d_model=len(le.classes_) + 2 + item_id_embedding_dim + item_category_embedding_dim + product_city_embedding_dim
,
    nhead=1,
    num_layers=2,
    dim_feedforward=512,
    dropout=0.1
)
def positional_encoding(seq_len, d_model, device):
    PE = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    PE = PE.to(device)
    return PE
    
class TransformerModel(nn.Module):
    def __init__(self, config, max_item_id, max_item_category, max_product_city, item_id_embedding_dim, item_category_embedding_dim, product_city_embedding_dim):
        super(TransformerModel, self).__init__()
        self.feature_size = config.feature_size
        self.d_model = config.d_model

        # Define the embedding layers for item_id, item_category, and product_city
        self.item_id_embedding = nn.Embedding(max_item_id, item_id_embedding_dim)
        self.item_category_embedding = nn.Embedding(max_item_category, item_category_embedding_dim)
        self.product_city_embedding = nn.Embedding(max_product_city, product_city_embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model)
        )
        self.output_layer = nn.Linear(config.d_model , config.feature_size)

    def forward(self, src, item_ids, item_categories, product_cities):
        # Compute the embeddings for item_ids, item_categories, and product_cities
        item_id_embeds = self.item_id_embedding(item_ids)
        item_category_embeds = self.item_category_embedding(item_categories)
        product_city_embeds = self.product_city_embedding(product_cities)

        # Concatenate the embeddings with the source tensor along the feature dimension
        src = torch.cat([src, item_id_embeds, item_category_embeds, product_city_embeds], dim=-1)
    
        current_seq_len = src.size(1)
        pos_encoder = positional_encoding(current_seq_len, self.d_model, device)
        src = src + pos_encoder
        transformed = self.transformer_encoder(src)
        output = self.output_layer(transformed[:, -1, :])
        return output

model = TransformerModel(config, max_item_id, max_item_category, max_product_city, 
                         item_id_embedding_dim, item_category_embedding_dim, 
                         product_city_embedding_dim)
#%%
class SessionDataset(Dataset):
    def __init__(self, X, y, item_ids, category_ids, product_cities):
        self.X = X
        self.y = y
        self.item_ids = item_ids
        self.category_ids = category_ids
        self.product_cities = product_cities

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.item_ids[index], self.category_ids[index], self.product_cities[index]

# Ensure target_indices is a 1D tensor with the same length as the first dimension of the other tensors
target_indices = target_indices.squeeze()

# Split your data
X_train, X_test, y_train, y_test, item_ids_train, item_ids_test, category_ids_train, category_ids_test, product_cities_train, product_cities_test = train_test_split(padded_input, target_indices, item_ids, category_ids, product_cities, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val, item_ids_train, item_ids_val, category_ids_train, category_ids_val, product_cities_train, product_cities_val = train_test_split(X_train, y_train, item_ids_train, category_ids_train, product_cities_train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

batch_size = 64
train_dataset = SessionDataset(X_train, y_train, item_ids_train, category_ids_train, product_cities_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = SessionDataset(X_val, y_val, item_ids_val, category_ids_val, product_cities_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = SessionDataset(X_test, y_test, item_ids_test, category_ids_test, product_cities_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%%
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

early_stopping = EarlyStopping(patience=10, delta=0.0)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        inputs, targets, item_ids, category_ids, product_cities = [tensor.to(device) for tensor in batch]
        optimizer.zero_grad()
        outputs = model(inputs, item_ids, category_ids, product_cities)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, targets, item_ids, category_ids, product_cities = [tensor.to(device) for tensor in batch]
            outputs = model(inputs, item_ids, category_ids, product_cities)
            loss = criterion(outputs, targets.long()) 
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    print(f"Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}")
    
    early_stopping(val_loss)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch in test_dataloader:
        inputs, targets, item_ids, category_ids, product_cities = [tensor.to(device) for tensor in batch]
        outputs = model(inputs, item_ids, category_ids, product_cities)
        loss = criterion(outputs, targets.long()) 


test_loss /= len(test_dataloader)
print(f"Test Loss: {test_loss:.4f}")
#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

model.eval()
all_outputs = []
all_targets = []

with torch.no_grad():
    for batch in test_dataloader:
        inputs, targets, item_ids, category_ids, product_cities = [tensor.to(device) for tensor in batch]
        outputs = model(inputs, item_ids, category_ids, product_cities)
        _, predicted = torch.max(outputs.data, 1)
        all_outputs.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

all_outputs = np.array(all_outputs)
all_targets = np.array(all_targets)

# Compute confusion matrix
cm = confusion_matrix(all_targets, all_outputs)

# After evaluation, for the confusion matrix
class_names = le.classes_
cm = confusion_matrix(all_targets, all_outputs)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Print the class each row is representing for
for i, class_name in enumerate(class_names):
    print(f"Row {i}: {class_name}")

# Compute precision, recall, f-score and support for each class
prfs = precision_recall_fscore_support(all_targets, all_outputs)

# Print precision, recall, f-score and support for each class
for i, class_name in enumerate(class_names):
    print(f"Class: {class_name}")
    print(f"Precision: {prfs[0][i]}")
    print(f"Recall: {prfs[1][i]}")
    print(f"F-score: {prfs[2][i]}")
    print(f"Support: {prfs[3][i]}")
    print("\n")


# %%
