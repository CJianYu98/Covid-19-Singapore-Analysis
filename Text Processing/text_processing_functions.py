import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import word_tokenize
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
    return re.sub(r"(?:\@|\#|https?\://)\S+", "", text)

def stopwords_ls(additional_stopwords):
    stop_list = stopwords.words('english')
    for word in additional_stopwords:
        stop_list.append(word)
    
    return stop_list

def text_preprocessing(df, column_name, stopword_list):
    output = []
    for text in df[column_name]:
        text = remove_hashtag_mentions_urls(text)
        text = remove_emoji(text)
        text_tokenize = word_tokenize(text)

        text_lower = [w.lower() for w in text_tokenize]
        text_words_only = [w for w in text_lower if re.search('^[a-z]+$',w)]
        text_stopremoved = [w for w in text_words_only if w not in stopword_list]

        stemmer = PorterStemmer() # can consider using other Stemmer
        text_stemmed = [stemmer.stem(w) for w in text_stopremoved]

        output.append(text_stemmed)

    return output

def demojize_text(df, column_name):
    emoji_decoded_tweets = []

    for text in df[column_name]:
        text = emoji.demojize(text)
        
        emoji_decoded_tweets.append(text)
    
    return emoji_decoded_tweets

def twitter_preprocessing(tweets, stopword_list):
    # drop useless columns
    tweets.drop(['quote_url', 'thumbnail', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'retweet_date', 'translate', 'trans_src', 'trans_dest', 'cashtags', 'timezone', 'created_at', 'retweet', 'near'], axis='columns', inplace=True)

    # remove those non english comments
    tweets_drop_non_en = tweets.drop(tweets[tweets.language != 'en'].index)

    tweets_drop_non_en['date'] = pd.to_datetime(tweets_drop_non_en['date'], infer_datetime_format=True)
    tweets_drop_non_en['time'] = pd.to_datetime(tweets_drop_non_en['time'], format= '%H:%M:%S').dt.time

    text_processed = text_preprocessing(tweets_drop_non_en, 'tweet', stopword_list)

    text_demojize = demojize_text(tweets_drop_non_en, 'tweet')

    tweets_drop_non_en['processed_text'] = text_processed
    tweets_drop_non_en['demojize_text'] = text_demojize

    return tweets_drop_non_en

def reddit_preprocessing(merged_df, stopword_list):

    # removing useless columns
    merged_df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)

    # renaming some columns for better readabilty
    merged_df.rename(columns={'created_dts_x': 'comment_created_dts', 'score_x':'comment_score', 'score_y':'post_score', 'created_dts_y':'post_created_dts'}, inplace=True)

    text_processed = text_preprocessing(merged_df, 'comment_body', stopword_list)

    text_demojize = demojize_text(merged_df, 'comment_body')

    merged_df['processed_text'] = text_processed
    merged_df['demojize_text'] = text_demojize

    return merged_df


def instagram_text_processing(comments, folder, filename):
    comments['processed_comment'] = None
    comments['emoji_decoded_comment'] = None

    for i in range(len(comments)):
        # getting the tweet at each row
        text = comments.iloc[i]['comment']

        # decoding emoji to text
        emoji_decoded_text = emoji.demojize(text)

        # removing mentions, hashtags, URLs 
        text = re.sub(r"(?:\@|\#|https?\://)\S+", "", text)
        #removing emoji
        text = remove_emoji(text)

        # tokenizing the text
        text_tokenize = word_tokenize(text)

        # changing text to lowercase, removing non words char, and removing stop words
        text_lower = [w.lower() for w in text_tokenize]
        text_words_only = [w for w in text_lower if re.search('^[a-z]+$',w)]
        stop_list = stopwords.words('english')
        text_stopremoved = [w for w in text_words_only if w not in stop_list]

        # perform stemming on the text
        stemmer = PorterStemmer()
        text_stemmed = [stemmer.stem(w) for w in text_stopremoved]

        # updating the cells to stored processed text
        comments.at[i,'processed_comment'] = text_stemmed
        comments.at[i,'emoji_decoded_comment'] = emoji_decoded_text

    # output the file as csv
    save_to_path = f'Instagram Data/{folder}/Cleaned Data/{filename}'
    comments.to_csv(save_to_path)