import os 
import json 
import pandas as pd 

def process_data(data_path):
    data = [] 
    for file in os.listdir(data_path):
        with open(os.path.join(data_path, file), 'r', encoding = 'utf-8') as f:
            data.extend(json.load(f)['chat'])
    
    df = pd.DataFrame(data)
    return df

def save_data(df, save_path):
    df.to_csv(save_path, index = False)

df = process_data('data')
df.to_csv('process/processed_data.csv', index = False)




