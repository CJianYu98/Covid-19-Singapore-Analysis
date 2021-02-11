import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from text_processing_functions import *


mypath = "."
folder_name='Twitter Data'
file_path = f'{mypath}/{folder_name}/'
files = [filename for filename in os.listdir(file_path) if filename.endswith('.csv')]

# text patterns for Emoji
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

for file in files:
    tweets = pd.read_csv(f'Twitter Data/{file}')

    index = file.index(' 010120')
    tweet_key_words = file[:index].split(' ')

    twitter_text_processing(tweets, tweet_key_words)