import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from text_processing_functions import *
import datetime


keywords = [] #
stopwords_list = stopwords_ls(keywords)

mypath = ".."
folder_name='Data/Reddit Data/Raw Data'
file_path = f'{mypath}/{folder_name}/'
folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store']

for folder in folders:
    folder_path = f'../Data/Reddit Data/Cleaned Data/{folder}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    filenames = [filename for filename in os.listdir(f'{file_path}{folder}') if filename.endswith('csv')]
    filenames.sort()

    for i in range(0, len(filenames), 2):
        comments = pd.read_csv(f'{file_path}{folder}/{filenames[i]}')
        posts = pd.read_csv(f'{file_path}{folder}/{filenames[i+1]}')

        merged_df = comments.merge(posts, left_on="comment_link_id", right_on="name")

        df = reddit_preprocessing(merged_df, stopwords_list)

        csv_name = filenames[i][7:-13]
        df.to_csv(f'{folder_path}/{csv_name}.csv')