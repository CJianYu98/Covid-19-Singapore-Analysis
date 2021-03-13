import os
import pandas as pd
from preprocess import *

file_path = f'../Data/Hardwarezone Data/Raw Data/New'
files = [file for file in os.listdir(file_path) if file != '.DS_Store']

keywords = [] #
stopwords_list = stopwords_ls(keywords)

for file in files:
    comments = pd.read_csv(f'{file_path}/{file}')

    df = hardwarezone_preprocessing(comments)

    df.to_csv(f'../Data/Hardwarezone Data/Cleaned Data/{file}')