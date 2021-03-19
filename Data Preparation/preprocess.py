import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from typing import Iterator
from sklearn.feature_extraction.text import CountVectorizer
import datetime as dt
import emoji
import pickle

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

def replace_characters(text: str) -> str:
    """
    Replace tricky punctuations that can mess up sentence tokenizers
    :param text: text with non-standard punctuations
    :return: text with standardized punctuations
    """
    replacement_rules = {'“': '"', '”': '"', '’': "'", '--': ','}
    for symbol, replacement in replacement_rules.items():
        text = text.replace(symbol, replacement)
    return text

def generate_tokenized_sentences(paragraph: str) -> Iterator[str]:
    """
    Tokenize each sentence in paragraph.
    For each sentence, tokenize each words and return the tokenized sentence one at a time.
    :param paragraph: text of paragraph
    """
    word_tokenizer = RegexpTokenizer(r'[-\'\w]+')

    for sentence in sent_tokenize(paragraph):
        tokenized_sentence = word_tokenizer.tokenize(sentence)
        if tokenized_sentence:
            tokenized_sentence.append('[END]')
            yield tokenized_sentence

def tokenize_raw_text(raw_text_path: str, token_text_path: str) -> None:
    """
    Read a input text file and write its content to an output text file in the form of tokenized sentences
    :param raw_text_path: path of raw input text file
    :param token_text_path: path of tokenized output text file
    """
    with open(raw_text_path) as read_handle, open(token_text_path, 'w') as write_handle:
        for paragraph in read_handle:
            paragraph = paragraph.lower()
            paragraph = replace_characters(paragraph)

            for tokenized_sentence in generate_tokenized_sentences(paragraph):
                write_handle.write(','.join(tokenized_sentence))
                write_handle.write('\n')

def get_tokenized_sentences(file_name: str) -> Iterator[str]:
    """
    Return tokenized sentence one at a time from a tokenized text
    :param file_name: path of tokenized text
    """
    # with open(file_name) as file_handle:
    #     for sentence in file_handle.read().splitlines():
    #         tokenized_sentence = sentence.split(',')
    #         yield tokenized_sentence

    for sent in file_name:
        tokenized_sentence = sent.split(',')
        yield tokenized_sentence

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

def facebook_preprocessing(comments):
    # removing useless columns
    comments = comments[['comments_raw', 'date_time']]

    # drop na comments
    comments.dropna(subset=['comments_raw'], inplace=True)

    # renaming some columns for better readabilty
    comments.rename(columns={'comments_raw': 'Comments', 'date_time': 'Datetime'}, inplace=True)

    # changing date/time to datetime format
    dates = []
    for row in comments['Datetime']:
        if '2020' not in row:
            row += ' 2021'
        if 'at' in row:
            try:
                date_time = dt.datetime.strptime(row, '%b %d at %I:%M %p %Y')
            except:
                pass
        elif '-' in row:
            date_time = dt.datetime.strptime(row, '%b-%d %Y')
        elif ',' in row: 
            date_time = dt.datetime.strptime(row, '%b %d, %Y')
        else:
            pass
        dates.append(date_time.date())
    comments['Datetime'] = dates

    comments_final = drop_short_comments(comments, 'Comments')

    return comments_final

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
    # file_path = '../Data/Instagram Data/(Final) Raw Data'
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
    # file_path = f'../Data/Hardwarezone Data/Raw Data/New'
    files = [file for file in os.listdir(file_path) if file != '.DS_Store']

    keywords = [] #
    stopwords_list = stopwords_ls(keywords)

    for file in files:
        comments = pd.read_csv(f'{file_path}/{file}')

        df = hardwarezone_preprocessing(comments)

        df.to_csv(f'../Data/Hardwarezone Data/Cleaned Data/{file}')

def process_facebook_text(file_path):
    # file_path = f'..Data/Facebook Data/Raw Data (with timestamp)'
    folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store' and folder != 'Old' and folder != 'readme.txt']

    keywords = [] #
    stopwords_list = stopwords_ls(keywords)

    for folder in folders:
        folder_path = f'../Data/Facebook Data/Cleaned Data/{folder}'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        policies = [file for file in os.listdir(f'{file_path}/{folder}') if file.endswith('.csv') and file != '.DS_Store']

        for policy in policies:
            df = pd.read_csv(f'{file_path}/{folder}/{policy}')

            df = facebook_preprocessing(df)

            df.to_csv(f'{folder_path}/{policy}')



actionable_keywords = 'should, shd, shld, may be, may b, maybe, mayb, to be, need, needs to, nids to, need to, nid to, believe, suppose to, ought, hope, have to, hav to, hv to, suggest, must, advise, request, require, better, btr, why cant, why cnt, how about, how bout, expect, please, pls, plz, why not, y not, why nt, y nt'
actionable_keywords = actionable_keywords.split(',')
for i in range(len(actionable_keywords)):
    actionable_keywords[i] = ' ' + actionable_keywords[i] + ' '

def label_actionable_comments(df, comment_header):    
    labels = []
    for row in df[comment_header]:
        for i, keyword in enumerate(actionable_keywords):
            if keyword in row:
                labels.append(1)
                break
            if i == len(actionable_keywords)-1:
                labels.append(0)

    df['actionable'] = labels

    return df

def get_actionable_comments(file_path, label = 1):
    df = pd.read_csv(file_path)
    actionable_comments = df[df['actionable'] == label]

    return actionable_comments

def train_vectorizer(train_df_path):
    train_df = pd.read_csv(train_df_path)
    sentences_train = train_df['Comment'].values
    vectorizer = CountVectorizer(stop_words='english', max_features=10000)
    return (vectorizer.fit(sentences_train))

vectorizer = train_vectorizer('/Users/chenjianyu/Library/Mobile Documents/com~apple~CloudDocs/SMU/SMU Module Materials/Y2S2/SMT203 Computational Social Sci/Covid-19-Singapore-Analysis/Data/Thoughtful Comments/thoughtful_comments_final(1).csv')

def label_valuable_comments(df, vectorizer):
    m1 = open("./valuable_classifiers/best_recall.pickle", 'rb')
    m2 = open("./valuable_classifiers/best_f1.pickle", 'rb')
    m3 = open("./valuable_classifiers/best_countvec_nofeatures.pickle", 'rb')
    clf1_load = pickle.load(m1)
    clf2_load = pickle.load(m2)
    clf3_load = pickle.load(m3)
    m1.close()
    m2.close()
    m3.close()

    sentences = df['Comments'].values

    X_test1 = df[['Num Pronouns', 'Average Loglikelihood', 'Relevance score']]
    X_test2 = df[['Num Pronouns', 'Average Loglikelihood', 'Length Category', 'Num Verbs', 'Num Discourse Relations']]
    X_test3 = vectorizer.transform(sentences)

    predictions1 = clf1_load.predict(X_test1)
    predictions2 = clf2_load.predict(X_test2)
    predictions3 = clf3_load.predict(X_test3)

    final_predictions = pd.DataFrame({"p1": predictions1, 'p2':predictions2, 'p3':predictions3})
    df['valuable'] = round((final_predictions['p1'] + final_predictions['p2'] + final_predictions['p3'])/3)

    return df

def get_thoughtful_comments(file_path, label = 1.0):
    df.pd.read_csv(file_path)
    valuable_comments = df[df['valuable'] == label]

    return valuable_comments