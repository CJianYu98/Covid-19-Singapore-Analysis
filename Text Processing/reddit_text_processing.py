import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from text_processing_functions import *
import datetime

reddit_posts_files = [
    'Reddit circuit breaker posts.csv',
    'Reddit vaccination posts.csv',
    'Reddit TraceTogether posts.csv'
]
reddit_comments_files = [
    'Reddit circuit breaker comments.csv',
    'Reddit vaccination comments.csv',
    'Reddit TraceTogether comments.csv'
]

mypath = "."
folder_name='Reddit Data'
file_path = f'{mypath}/{folder_name}/'

for i in range(len(reddit_posts_files)):

    reddit_post = pd.read_csv(file_path + reddit_posts_files[i])
    reddit_comments = pd.read_csv(file_path + reddit_comments_files[i])

    comments_merged = reddit_comments.merge(reddit_post, left_on="comment_link_id", right_on="name")

    reddit_text_processing(comments_merged, reddit_comments_files[i])