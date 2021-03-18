import os
import datetime
# Import Data Processing and Visualization packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import Text Processing packages
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob


def get_num_comments(folder_path):
    files = [file for file in os.listdir(folder_path) if file != '.DS_Store']

    num_comments = 0
    for file in files:
        df = pd.read_csv(f'{folder_path}/{file}')
        num_comments += len(df)
    
    return num_comments

def get_vader_sentiment(file_path, comment_header):
    df = pd.read_csv(file_path)
    vader_analyser = SentimentIntensityAnalyzer()

    vader_neg_sentiment = []
    vader_neu_sentiment = []
    vader_pos_sentiment = []
    vader_compound_sentiment = []

    for row in df[comment_header]:
        score = vader_analyser.polarity_scores(row)

        vader_neg_sentiment.append(score['neg'])
        vader_neu_sentiment.append(score['neu'])
        vader_pos_sentiment.append(score['pos'])
        vader_compound_sentiment.append(score['compound'])
    
    df['Vader_neg_score'] = vader_neg_sentiment
    df['Vader_neu_score'] = vader_neu_sentiment
    df['Vader_pos_score'] = vader_pos_sentiment
    df['Vader_compound_score'] = vader_compound_sentiment

    return df

def get_textblob_sentiment(file_path, comment_header):
    
    df = pd.read_csv(file_path)

    polarity_scores = []
    subjectivity_scores = []

    for row in df[comment_header]:
        analysis = TextBlob(row)
        polarity_scores.append(analysis.sentiment.polarity)
        subjectivity_scores.append(analysis.sentiment.subjectivity)
    
    df['Textblob_polarity_score'] = polarity_scores
    df['Textblob_subjectivity_score'] = subjectivity_scores

    return df


def merge_policy_df(policy):
    path_parent = "/Users/chenjianyu/Desktop/Y2S2/SMT203 Computational Social Sci" #change to your own path for whoever running

    reddit_folder = f"{path_parent}/Covid-19-Singapore-Analysis/Data/Reddit Data/Cleaned Data/Policies/Combined"
    reddit_files = [file for file in os.listdir(reddit_folder) if file != '.DS_Store']
    for file in reddit_files:
        if policy in file:
            reddit_policy_file = file
    reddit_df = pd.read_csv(f'{reddit_folder}/{file}')
    reddit_df = reddit_df[['comment_body', 'comment_created_dts']]
    reddit_df.rename({"comment_body": 'text', 'comment_created_dts':'datetime'}, inplace=True)

    insta_folder = f"{path_parent}/Covid-19-Singapore-Analysis/Data/Instagram Data/Cleaned Data/Policies/Combined"
    reddit_files = [file for file in os.listdir(reddit_folder) if file != '.DS_Store']
    for file in reddit_files:
        if policy in file:
            reddit_policy_file = file
    reddit_df = pd.read_csv(f'{reddit_folder}/{file}')
    reddit_df = reddit_df[['comment_body', 'comment_created_dts']]
    reddit_df.rename({"comment_body": 'text', 'comment_created_dts':'datetime'}, inplace=True)





    final_df = pd.concat([reddit_df, insta_df.....])



merge_policy_df(circuit_breaker)
