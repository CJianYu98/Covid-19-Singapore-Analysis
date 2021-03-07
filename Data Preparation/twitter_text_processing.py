import os
import pandas as pd
from preprocess import *

file_path = f'../Data/Twitter Data/Raw Data'
folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store']

keywords = [] #
stopwords_list = stopwords_ls(keywords)

for f in folders:
    policies_folders = [f_name for f_name in os.listdir(f'{file_path}/{f}') if f_name != '.DS_Store']

    f_path = f'../Data/Twitter Data/Cleaned Data/{f}'
    if not os.path.exists(f_path):
        os.mkdir(f_path)
    
    for folder in policies_folders:
        files = [filename for filename in os.listdir(f'{file_path}/{f}/{folder}') if filename.endswith('.csv')]

        folder_path = f'../Data/Twitter Data/Cleaned Data/{f}/{folder}'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        for file in files:
            tweets = pd.read_csv(f'{file_path}/{f}/{folder}/{file}')

            df = twitter_preprocessing(tweets)

            df.to_csv(f'{folder_path}/{file}')

