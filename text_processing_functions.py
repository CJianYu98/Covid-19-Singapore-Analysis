import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import datetime

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


def twitter_text_processing(tweets, tweet_key_words):
    # drop useless columns
    tweets.drop(['quote_url', 'thumbnail', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'retweet_date', 'translate', 'trans_src', 'trans_dest', 'cashtags', 'timezone', 'created_at', 'retweet', 'near'], axis='columns', inplace=True)

    # remove those non english comments
    tweets_drop_non_en = tweets.drop(tweets[tweets.language != 'en'].index)
    
    # add another column to store processed tweet text
    tweets_drop_non_en['processed_tweets'] = None

    # creating stop word list, added the search key words in it cause every comment will have them. 
    stop_list = stopwords.words('english')
    for word in tweet_key_words:
        stop_list.append(word)

    for i in range(len(tweets_drop_non_en)):
        # getting the tweet at each row
        text = tweets_drop_non_en.iloc[i]['tweet']

        # removing mentions, hashtags, URLs from tweet
        text = re.sub(r"(?:\@|\#|https?\://)\S+", "", text)
        #removing emoji
        text = remove_emoji(text)

        # tokenizing the text
        text_tokenize = word_tokenize(text)

        # changing text to lowercase, removing non words char, and removing stop words
        text_lower = [w.lower() for w in text_tokenize]
        text_words_only = [w for w in text_lower if re.search('^[a-z]+$',w)]
        text_stopremoved = [w for w in text_words_only if w not in stop_list]

        # perform stemming on the text
        stemmer = PorterStemmer()
        text_stemmed = [stemmer.stem(w) for w in text_stopremoved]

        # adding the processed text back into the specific row and column
        try:
            tweets_drop_non_en.at[i,'processed_tweets'] = text_stemmed
        except:
            pass
    
    # drop those rows with many NAs, dk why it will appear after tokenizing lol
    tweets_drop_non_en.dropna(thresh=17, inplace=True)

    # converting date time columns to datetime type
    tweets_drop_non_en['date'] = pd.to_datetime(tweets_drop_non_en['date'], infer_datetime_format=True)
    tweets_drop_non_en['time'] = pd.to_datetime(tweets_drop_non_en['time'], format= '%H:%M:%S').dt.time

    # exporting to csv file
    key_words = "_".join(tweet_key_words)
    tweets_drop_non_en.to_csv(f'Twitter Data/Cleaned Data/{key_words}.csv')


def reddit_text_processing(comments_merged, filename):

    # removing useless columns
    comments_merged.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)

    # renaming some columns for better readabilty
    comments_merged.rename(columns={'created_dts_x': 'comment_created_dts', 'score_x':'comment_score', 'score_y':'post_score', 'created_dts_y':'post_created_dts'}, inplace=True)

    # add a new column to store processed reddit comments
    comments_merged['processed_comment'] = None

    for i in range(len(comments_merged)):
        # getting the tweet at each row
        text = comments_merged.iloc[i]['comment_body']

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
        comments_merged.at[i,'processed_comment'] = text_stemmed

    # output the file as csv
    save_to_path = 'Reddit Data/Cleaned Data/' + filename
    comments_merged.to_csv(save_to_path)


def instagram_text_processing(comments, folder, filename):
    comments['processed_comment'] = None

    for i in range(len(comments)):
        # getting the tweet at each row
        text = comments.iloc[i]['comment']

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

    # output the file as csv
    save_to_path = f'Instagram Data/{folder}/Cleaned Data/{filename}'
    comments.to_csv(save_to_path)