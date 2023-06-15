# Master-Thesis-OnFliggy
## Dataset
Available at https://tianchi.aliyun.com/dataset/113649

Fliggy dataset contains three subsets: 
- user_behavior_transfer_plans.csv which contains all user behavior data, including User ID, product ID, behavior type, timestamp
- user_profile.csv all user basic attribute portraits, including User ID, age, gender, occupation, habitual city, crowd tag
- transfer_plan_profile.csv all the basic attributes of the product, including Product ID, product category ID, product city, product tag

## Problem
The thesis aims to predict the last behavior of the user within the sequence mainly based on the behavior history data, comparing with utilizing embedded item information altogether. 

## Model Used
Deep learning models including LSTM and Transformer model. 
- LSTM many to one -- predicting the last behavior based on only the behavior history.
- LSTM embedded -- predicting based on behavior history and embedded item information.
- Transformer many to one -- predicting the last behavior based on only the behavior history.
- Transformer embedded -- includes embedded item information. 

Extra features are included: daytime or not and weekday or not. 

