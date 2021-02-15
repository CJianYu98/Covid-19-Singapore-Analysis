import os
import pandas as pd
from text_processing_functions import *

mypath = ".."
folder_name='Data/Twitter Data/Raw Data'
file_path = f'{mypath}/{folder_name}/'
folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store']

keywords = [] #
stopwords_list = stopwords_ls(keywords)

for f in folders:
    keywords_folders = [f_name for f_name in os.listdir(f'{file_path}{f}') if f_name != '.DS_Store']

    folder_path = f'../Data/Twitter Data/Cleaned Data/{f}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    for folder in keywords_folders:
        files = [filename for filename in os.listdir(f'{file_path}{f}/{folder}') if filename.endswith('.csv')]

        datasets = []

        for file in files:
            tweets = pd.read_csv(f'{file_path}{f}/{folder}/{file}')

            df = twitter_preprocessing(tweets, stopwords_list)

            datasets.append(df)
        
        final_df = pd.concat(datasets)

        final_df.to_csv(f'{folder_path}/{folder}.csv')

