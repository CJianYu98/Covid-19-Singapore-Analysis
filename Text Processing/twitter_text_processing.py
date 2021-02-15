import os
from text_processing_functions import *

mypath = ".."
folder_name='Data/Twitter Data/Raw Data'
file_path = f'{mypath}/{folder_name}/'
folders = [folder for folder in os.listdir(file_path)]

for f in folders:
    keywords_folders = [f_name for f_name in os.listdir(f'{file_path}{f}')]

    folder_path = f'../Data/Twitter Data/Cleaned Data/{f}'
    os.mkdir(folder_path)
    
    
    for folder in folders:
        files = [filename for filename in os.listdir(f'file_path{f}/{folder}') if filename.endswith('.csv')]

        datasets = []

        for file in files:
            tweets = pd.read_csv(f'{file_path}{f}/{folder}/{file}')

            key_words = ['vaccine', 'vaccination', 'singapore']
            stopwords_list = stopwords_ls(key_words)

            df = twitter_preprocessing(tweets, stopwords_list)

            datasets.append(df)
        
        final_df = pd.concat[datsets]


        

        final_df.to_csv(f'{folder_path}/{}')




        index = file.index(' 010120')
        key_words = file[:index].split(' ')

        twitter_preprocessing(tweets, tweet_key_words)
