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

def get_actionable_comments(df, label = 1):
    # df = pd.read_csv(file_path)
    actionable_comments = df[df['actionable'] == label]

    return actionable_comments

def get_valuable_comments(df, label = 1.0):
    # df = pd.read_csv(file_path)
    valuable_comments = df[df['valuable'] == label]

    return valuable_comments

def get_merged_policy_df(policy):
    df_list = []
    path_parent = "C:/Users/user/Documents/GitHub" # Change main path accordingly

    # Reddit
    reddit_folder = f"{path_parent}/Covid-19-Singapore-Analysis/Data/Reddit Data/Cleaned Data/Policies/Combined"
    reddit_files = [file for file in os.listdir(reddit_folder) if file != '.DS_Store']
    for file in reddit_files:
        if policy in file.lower():
            reddit_policy_file = file
            reddit_df = pd.read_csv(f'{reddit_folder}/{reddit_policy_file}')
            reddit_df = reddit_df[['Comments', 'Comment Datetime', 'valuable']]
            reddit_df['platform'] = 'reddit'
            # reddit_df.rename(columns={"comment_body": 'text', 'comment_created_dts':'datetime'}, inplace=True)
            df_list.append(reddit_df)
            break 

    # Twitter 
    twitter_folder = f"{path_parent}/Covid-19-Singapore-Analysis/Data/Twitter Data/Cleaned Data/Policies/Combined"
    twitter_files = [file for file in os.listdir(twitter_folder) if file != '.DS_Store']
    for file in twitter_files:
        if policy in file.lower():
            twitter_policy_file = file
            twitter_df = pd.read_csv(f'{twitter_folder}/{twitter_policy_file}')
            # twitter_df["datetime"] = twitter_df["Comment Datetime"] + " " + twitter_df["time"]
            twitter_df = twitter_df[['Comments', 'Comment Datetime', 'valuable']]
            twitter_df['platform'] = 'twitter'
            # twitter_df.rename(columns={"tweet": 'text'}, inplace=True)
            df_list.append(twitter_df)
            break

    # Instagram
    insta_folder = f"{path_parent}/Covid-19-Singapore-Analysis/Data/Instagram Data/Cleaned Data/Policies/Combined"
    insta_files = [file for file in os.listdir(insta_folder) if file != '.DS_Store']
    for file in insta_files:
        if policy in file:
            insta_policy_file = file
            insta_df = pd.read_csv(f'{insta_folder}/{insta_policy_file}')
            insta_df = insta_df[['Comments', 'Comment Datetime', 'valuable']]
            insta_df['platform'] = 'instagram'
            # insta_df.rename(columns = {"Comments": 'text', 'Comment Datetime':'datetime'}, inplace=True)
            df_list.append(insta_df)
            break

    # Hardwarezone 
    hwz_folder = f"{path_parent}/Covid-19-Singapore-Analysis/Data/Hardwarezone Data/Cleaned Data"
    hwz_files = [file for file in os.listdir(hwz_folder) if file != '.DS_Store']
    for file in hwz_files:
        if policy in file:
            hwz_policy_file = file
            hwz_df = pd.read_csv(f'{hwz_folder}/{hwz_policy_file}')
            hwz_df = hwz_df[['Comments', 'Comment Datetime', 'valuable']]
            hwz_df['platform'] = 'hardwarezone'
            # hwz_df.rename(columns = {"Comments": 'text', 'Thread Datetime':'datetime'}, inplace=True)
            df_list.append(hwz_df)
            break

    # Facebook 
    facebook_folder = f"{path_parent}/Covid-19-Singapore-Analysis/Data/Facebook Data/Cleaned Data/Policies/Combined" # To change
    facebook_files = [file for file in os.listdir(facebook_folder) if file != '.DS_Store']
    for file in facebook_files:
        if policy in file:
            facebook_policy_file = file
            facebook_df = pd.read_csv(f'{facebook_folder}/{facebook_policy_file}')
            facebook_df = facebook_df[['Comments', 'Comment Datetime', 'valuable']]
            facebook_df['platform'] = 'facebook'
            # facebook_df.rename(columns = {"Comments": 'text', 'Comment Datetime':'datetime'}, inplace=True)
            df_list.append(facebook_df)
            break
            
    final_df = pd.concat(df_list)
    return final_df
