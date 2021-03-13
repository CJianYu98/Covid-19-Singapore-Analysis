import os
import pandas as pd
from preprocess import *


file_path = '../Data/Instagram Data/(Final) Raw Data'
insta_acc_folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store']

keywords = [] #
stopwords_list = stopwords_ls(keywords)

"""
    Code for merging instagram excel files into one based on the same keyword search at the same Instagram Acc. Un-comment to run if needed. 
"""
for insta_acc in insta_acc_folders:
    folder_path = f'../Data/Instagram Data/Cleaned Data/Final/{insta_acc}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    policies = [file for file in os.listdir(f'{file_path}/{insta_acc}') if file != '.DS_Store']
    
    for policy in policies:
        df = pd.read_csv(f'{file_path}/{insta_acc}/{policy}')

        df = instagram_preprocessing(df)

        df.to_csv(f'{folder_path}/{policy}')
