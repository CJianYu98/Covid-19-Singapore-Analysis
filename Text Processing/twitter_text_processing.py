import os
from text_processing_functions import *

mypath = "."
folder_name='Twitter Data'
file_path = f'{mypath}/{folder_name}/'
files = [filename for filename in os.listdir(file_path) if filename.endswith('.csv')]


for file in files:
    tweets = pd.read_csv(f'Twitter Data/{file}')

    index = file.index(' 010120')
    tweet_key_words = file[:index].split(' ')

    twitter_text_processing(tweets, tweet_key_words)