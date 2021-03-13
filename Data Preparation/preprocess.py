import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
import datetime
import emoji

def remove_emoji(text):
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)

def remove_hashtag_mentions_urls(text):
    # e.sub(r"(?:\@|https?\://)\S+", "", text) # remove mentions and url only
    return re.sub(r"(?:\@|\#|https?\://)\S+", "", text)

def stopwords_ls(additional_stopwords):

    stop_list = stopwords.words('english')
    for word in additional_stopwords:
        stop_list.append(word)
    
    return stop_list

def demojize_text(df, column_name):
    emoji_decoded_tweets = []

    for text in df[column_name]:
        text = emoji.demojize(text)
        
        emoji_decoded_tweets.append(text)
    
    return emoji_decoded_tweets

def drop_short_comments(df, text_column):

    text_length = []
    for text in df[text_column]:
        text = remove_hashtag_mentions_urls(text)
        text_tokenize = TweetTokenizer().tokenize(text)

        text_lower = [w.lower() for w in text_tokenize]
        text_words_only = [w for w in text_lower if re.search('^[a-z]+$',w)]
        text_length.append(len(text_words_only))
        
    df['num_words'] = text_length #Creating feature 1 for thoughtful comments
    df_final = df.drop(df[df.num_words < 5].index)    

    return df_final

def twitter_preprocessing(tweets):
    # remove those non english comments
    tweets_en = tweets.drop(tweets[tweets.language != 'en'].index)

    # drop useless columns
    tweets_en = tweets_en[['id', 'date', 'time', 'tweet', 'hashtags']]

    tweets_en['date'] = pd.to_datetime(tweets_en['date'], infer_datetime_format=True)
    tweets_en['time'] = pd.to_datetime(tweets_en['time'], format= '%H:%M:%S').dt.time
    
    tweets_final = drop_short_comments(tweets_en, 'tweet')

    return tweets_final

def reddit_preprocessing(merged_df):
    # removing useless columns
    merged_df = merged_df[['title', 'created_dts_y', 'comment_body', 'created_dts_x']]

    # renaming some columns for better readabilty
    merged_df.rename(columns={'created_dts_x': 'comment_created_dts', 'created_dts_y':'post_created_dts'}, inplace=True)

    final_df = drop_short_comments(merged_df, 'comment_body')

    return final_df

def instagram_preprocessing(comments):
    # removing useless columns
    comments = comments[['description', 'pubDate','comments', 'timestamp', ]]

    # renaming some columns for better readabilty
    comments.rename(columns={'description': 'Post Description', 'pubDate':'Post Datetime', 'comments': 'Comments', 'timestamp':'Comment Datetime'}, inplace=True)

    # changing date/time to datetime format
    comments['Comment Datetime'] = pd.to_datetime(comments['Comment Datetime'], infer_datetime_format=True)
    comments['Post Datetime'] = pd.to_datetime(comments['Post Datetime'], infer_datetime_format=True)

    comments_final = drop_short_comments(comments, 'Comments')

    return comments_final

def hardwarezone_preprocessing(comments):
    # removing useless columns
    comments = comments[['thread', 'datetime', 'post_text', 'post_timestamp']]

    # drop na comments
    comments.dropna(subset=['post_text'], inplace=True)

    # renaming some columns for better readabilty
    comments.rename(columns={'thread': 'Threads', 'datetime': 'Thread Datetime', 'post_text': 'Comments', 'post_timestamp': 'Comment Datetime'}, inplace=True)

    # changing date/time to datetime format
    comments['Thread Datetime'] = pd.to_datetime(comments['Thread Datetime'], infer_datetime_format=True)
    comments['Comment Datetime'] = pd.to_datetime(comments['Comment Datetime'], infer_datetime_format=True)

    comments_final = drop_short_comments(comments, 'Comments')

    return comments_final

def facebook_preprocessing(df, stopword_list):

    df.dropna(subset=['Comment'], inplace=True)

    text_processed = text_preprocessing(df, 'Comment', stopword_list)

    text_demojize = demojize_text(df, 'Comment')

    df['processed_text'] = text_processed
    df['demojize_text'] = text_demojize

    df.dropna(subset=['processed_text'], inplace=True)

    return df 


actionable_keywords = 'should, shd, shld, may be, may b, maybe, mayb, to be, needs to, nids to, need to, nid to, believe, suppose to, ought, hope, have to, hav to, hv to, suggest, must, advise, request, require, better, btr, why cant, why cnt, how about, how bout, expect, please, pls, plz, why not, y not, why nt, y nt'
actionable_keywords = actionable_keywords.split(',')
for i in range(len(actionable_keywords)):
    actionable_keywords[i] = ' ' + actionable_keywords[i] + ' '

def label_actionable_comments(df, comment_header):
    labels = []
    for row in df[comment_header]:
        for keyword in actionable_keywords:
            if keyword in row:
                labels.append(1)
            else:
                labels.append(0)

    df[actionable_comment] = labels

    return df


