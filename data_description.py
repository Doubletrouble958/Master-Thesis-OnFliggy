# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
 #%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
#%%
reader = pd.read_csv('user_item_behavior_history.csv',header=None,names=['userid','itemid','behavior','timestamp'],iterator=True)

loop = True
chunkSize = 10000000
chunks = []

import datetime
starttime = datetime.datetime.now()
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped")

df = pd.concat(chunks, ignore_index=True)
endtime = datetime.datetime.now()

print('loop_time:',(endtime - starttime).seconds)
#%%
print(df.shape)
#%%
num_sample = int(0.01*df['userid'].nunique())
unique_users_sample = df['userid'].drop_duplicates().sample(num_sample,random_state=42)
df=df[df['userid'].isin(unique_users_sample)]
#%%
df2 = pd.read_csv('item_profile.csv',header=None,names=['itemid', 'categoryid', 'product city', 'product tag'])
#%%
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
#%%
df = pd.merge(df, df2, on='itemid')
# Remove unnecessary columns
df = df.drop(['product tag'], axis=1)
df = df.sort_values(by=['userid', 'timestamp'])
df= df.reset_index(drop=True)
#%%
def process_dataframe(df):
    df['index'] = -1
    current_index = 0
    user_indexes = {}

    for i, row in df.iterrows():
        user_id = row['userid']
        timestamp = row['timestamp']

        if user_id not in user_indexes:
            user_indexes[user_id] = {'index': current_index, 'timestamp': timestamp}
            df.at[i, 'index'] = current_index
            current_index += 1
        else:
            if timestamp - user_indexes[user_id]['timestamp'] <= timedelta(hours=1):
                df.at[i, 'index'] = user_indexes[user_id]['index']
            else:
                user_indexes[user_id]['index'] = current_index
                user_indexes[user_id]['timestamp'] = timestamp
                df.at[i, 'index'] = user_indexes[user_id]['index']
                current_index += 1

    return df
#%%
# Process the DataFrame
result = process_dataframe(df)
#%%
# Specify the filename and path where you want to save the CSV file
filename = 'simdata.csv'
result.to_csv(filename, index=False)
#%%
def draw_pie_chart(df):
    index_counts = df['index'].value_counts()
    
    group1 = sum(index_counts < 5)
    group2 = sum((index_counts >= 5) & (index_counts <= 10))
    group3 = sum(index_counts > 10)

    labels = ['Less than 5', 'Between 5 and 10', 'More than 10']
    sizes = [group1, group2, group3]
    colors = ['#66b3ff', '#99ff99', '#ff9999']
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')

    plt.title("Number of actions in the same session")
    plt.show()

# Call the function with the result DataFrame
draw_pie_chart(result)
#%%
def draw_unique_itemid_pie_chart(df):
    unique_itemid_counts = df.groupby('index')['itemid'].nunique()

    group1 = sum(unique_itemid_counts < 5)
    group2 = sum(unique_itemid_counts >= 5)
                 
    labels = ['Less than 5', 'More than 5']
    sizes = [group1, group2]
    colors = ['#66b3ff', '#99ff99']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')

    plt.title("Number of items for each index")
    plt.show()

# Call the function with the result DataFrame
draw_unique_itemid_pie_chart(result)
#%%
unique_user_id = df['userid'].nunique()
unique_item_id = df['itemid'].nunique()
unique_category_id = df['categoryid'].nunique()
unique_product_city = df['product city'].nunique()
print("Number of unique user id:", unique_user_id)
print("Number of unique item id:", unique_item_id)
print("Number of unique category id:", unique_category_id)
print("Number of unique product city:", unique_product_city)