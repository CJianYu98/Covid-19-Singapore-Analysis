import os
import pandas as pd
from preprocess import *

def process_twitter_text(file_path):
    # file_path = f'../Data/Twitter Data/Raw Data'
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

def process_reddit_text(file_path): 
    # file_path = f'../Data/Reddit Data/Raw Data'
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

def process_insta_text(file_path):
    file_path = '../Data/Instagram Data/(Final) Raw Data'
    insta_acc_folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store']

    keywords = [] #
    stopwords_list = stopwords_ls(keywords)

    for insta_acc in insta_acc_folders:
        folder_path = f'../Data/Instagram Data/Cleaned Data/Final/{insta_acc}'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        policies = [file for file in os.listdir(f'{file_path}/{insta_acc}') if file != '.DS_Store']
        
        for policy in policies:
            df = pd.read_csv(f'{file_path}/{insta_acc}/{policy}')

            df = instagram_preprocessing(df)

            df.to_csv(f'{folder_path}/{policy}')

def process_hwz_text(file_path):
    file_path = f'../Data/Hardwarezone Data/Raw Data/New'
    files = [file for file in os.listdir(file_path) if file != '.DS_Store']

    keywords = [] #
    stopwords_list = stopwords_ls(keywords)

    for file in files:
        comments = pd.read_csv(f'{file_path}/{file}')

        df = hardwarezone_preprocessing(comments)

        df.to_csv(f'../Data/Hardwarezone Data/Cleaned Data/{file}')

def process_facebook_text(file_path):
    mypath = ".."
    folder_name='Data/Facebook Data/Raw Data'
    file_path = f'{mypath}/{folder_name}/fb_merged_data.csv'
    # folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store']

    keywords = [] #
    stopwords_list = stopwords_ls(keywords)

    df = pd.read_csv(file_path)

    df = facebook_text_processing(df, stopwords_list)

    df.to_csv('../Data/Facebook Data/Cleaned Data/fb_merged_data.csv')

