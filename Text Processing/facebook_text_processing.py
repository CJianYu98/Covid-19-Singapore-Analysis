import os
import pandas as pd
from text_processing_functions import *

mypath = ".."
folder_name='Data/Facebook Data/Raw Data'
file_path = f'{mypath}/{folder_name}/fb_merged_data.csv'
# folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store']

keywords = [] #
stopwords_list = stopwords_ls(keywords)

df = pd.read_csv(file_path)

df = facebook_text_processing(df, stopwords_list)

df.to_csv('../Data/Facebook Data/Cleaned Data/fb_merged_data.csv')