import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from preprocess import *
import datetime

file_path = f'../Data/Reddit Data/Raw Data'
folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store']

keywords = [] #
stopwords_list = stopwords_ls(keywords)

for f in folders:
    policies_folders = [f_name for f_name in os.listdir(f'{file_path}/{f}') if f_name != '.DS_Store']

    f_path = f'../Data/Reddit Data/Cleaned Data/{f}'
    if not os.path.exists(f_path):
        os.mkdir(f_path)
    
    for folder in policies_folders:
        files = [filename for filename in os.listdir(f'{file_path}/{f}/{folder}') if filename.endswith('.csv')]
        files.sort()

        folder_path = f'../Data/Reddit Data/Cleaned Data/{f}/{folder}'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        for i in range(0, len(files), 2):
            comments = pd.read_csv(f'{file_path}/{f}/{folder}/{files[i]}')
            posts = pd.read_csv(f'{file_path}/{f}/{folder}/{files[i+1]}')

            merged_df = comments.merge(posts, left_on="comment_link_id", right_on="name")

            df = reddit_preprocessing(merged_df)

            csv_name = files[i][7:-13]
            df.to_csv(f'{folder_path}/{csv_name}.csv')