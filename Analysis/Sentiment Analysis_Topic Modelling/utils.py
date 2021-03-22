import os
import datetime
# Import Data Processing and Visualization packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import Text Processing packages
import re
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import gensim
import pyLDAvis.gensim
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def get_num_comments(folder_path):
    files = [file for file in os.listdir(folder_path) if file != '.DS_Store']

    num_comments = 0
    for file in files:
        df = pd.read_csv(f'{folder_path}/{file}')
        num_comments += len(df)
    
    return num_comments

def get_vader_sentiment(df, comment_header):
    # df = pd.read_csv(file_path)
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

def get_textblob_sentiment(df, comment_header):
    
    # df = pd.read_csv(file_path)

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

def get_policy_data(policy, folders):
    frames = []
    for folder in folders:
        files = [file for file in os.listdir(folder) if file.endswith('.csv')]
        for file in files:
            if policy.lower() in file.lower():
                df = pd.read_csv(f'{folder}/{file}')
                df = df[['Comments', 'Comment Datetime', 'actionable', 'valuable']]
                frames.append(df)
                print(True)
                break
    final_df = pd.concat(frames, ignore_index=True)
    return final_df


pos_tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS']
# 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
def corpus2docs(df):
    docs1 = [TweetTokenizer().tokenize(comment) for comment in df['Comments']]
    for i, comment in enumerate(docs1):
        tags = nltk.pos_tag(comment)
        docs_tags = [tag[0].lower() for tag in tags if tag[1] in pos_tags]
        docs1[i] = docs_tags
    # docs2 = [[w.lower() for w in doc] for doc in docs1]
    docs3 = [[w for w in doc if re.search('^[a-z]+$', w)] for doc in docs1]
    docs4 = [[w for w in doc if w not in stop_list] for doc in docs3]
    return docs4

def docs2vecs(docs, dic):
    vecs = [dic.doc2bow(doc) for doc in docs]
    return vecs

stop_list = nltk.corpus.stopwords.words('english')

def topics_visualization_gensim(vecs, dic, num_topics, topic, top_words):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=vecs, id2word=dic, num_topics=num_topics, random_state=99, iterations=100)
    pyLDAvis.enable_notebook()
    visual= pyLDAvis.gensim.prepare(lda_model, vecs, dic)
    pyLDAvis.save_html(visual, f"{topic}_viz.html")
    return lda_model.show_topics(num_topics, top_words)
