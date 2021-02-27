import pandas as pd
import re
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import math
from unigram import *
from preprocess import *
# import sys
# sys.path.insert(0, '/Users/chenjianyu/Desktop/Y2S2/SMT203 Computational Social Sci/Covid-19-Singapore-Analysis/Data Preparation')
# from text_processing_functions import *

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


# Thoughtful comment feature 1
def get_comment_length(text: str) -> int:
    """
    Count the number of words in a comment. Feature 1 for thoughtful comment (Structural feature). 

    Args:
        text (str): Comment of a user

    Returns:
        int: Number of words in a comment
    """
    text = replace_characters(text)
    text = remove_hashtag_mentions_urls(text)
    text = remove_emoji(text)

    word_tokenizer = RegexpTokenizer(r'[-\'\w]+')
    tokenized_text = word_tokenizer.tokenize(text)
    return len(tokenized_text)

# Thoughtful comment feature 2
def comment_likelihood(text: str, train_model: UnigramModel) -> float:
    """
    Calculate the average loglikelihood between a comment and a well stuctured news text from their respective unigram models. Feature 2 for thoughtful comment (Lexical feature).

    Args:
        text (str): Comment of a user
        train_model (UnigramModel): Unigram model object of the well structured news text

    Returns:
        float: Average loglikelihood score
    """
    text_replaced = replace_characters(text)
    txt = ''

    for tokenized_sentence in generate_tokenized_sentences(text_replaced):
        sent = ','.join(tokenized_sentence)
        txt += sent

    text_counter = UnigramCounter(txt)

    text_avg_log_likelihood = train_model.evaluate(text_counter)
    return text_avg_log_likelihood


def news_articles_unigram(file_name: str) -> UnigramModel:
    """
    Creating a unigram model for news article by reputatable news sources. This unigram model will be used to calculate average loglikelihood for an user's comment.

    Args:
        file_name (str): The csv file which contains all the news aricles

    Returns:
        UnigramModel: An unigram object
    """
    df = pd.read_csv(file_name)
    df = df[['content','id', 'publication']]

    corpus = []

    for i, row in df.iterrows():
        article = row['content']

        article_replaced = replace_characters(article)

        for tokenized_sentence in generate_tokenized_sentences(article_replaced):
            s = ','.join(tokenized_sentence)
            corpus.append(s)
    
    train_counter = UnigramCounter(corpus)

    train_model = UnigramModel(train_counter)
    train_model.train(k=1)

    return train_model


# Thoughtful comment feature 3
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # POS tags for verbs 

def get_num_verbs(text: str) -> int:
    """
    Count the number of verbs in a comment based on their POS tagging. Feature 3 for thoughtful comment (Syntatic Feature).

    Args:
        text (str): Comment of an user

    Returns:
        int: Number of verbs in a comment
    """

    text = replace_characters(text)
    text = remove_hashtag_mentions_urls(text)
    text = remove_emoji(text)

    word_tokenizer = RegexpTokenizer(r'[-\'\w]+')
    tokenized_text = word_tokenizer.tokenize(text)

    text_tags = nltk.pos_tag(tokenized_text)

    count = 0
    for tag in text_tags:
        if tag[1] in verb_tags:
            count += 1
    
    return count


# Thoughtful comment feature 4
discourse_keywords = [' what’s more as a matter of fact ', ' at least partly because ', ' on the one hand indeed ', ' in large part because ', ' particularly because ', ' largely as a result ', ' particularly since ', ' especially because ', ' on the other hand ', ' primarily because ', ' at the same time ', ' especially since ', ' not only because ', ' reportedly after ', ' on the contrary ', ' in part because ', ' largely because ', ' particularly as ', ' perhaps because ', ' particularly if ', ' on the one hand ', ' simply because ', ' merely because ', ' in other words ', ' partly because ', ' mainly because ', ' in particular ', ' for one thing ', ' as if besides ', ' one day after ', ' especially as ', ' by comparison ', ' especially if ', ' also although ', ' consequently ', ' on the whole ', ' when if only ', ' particularly ', ' incidentally ', ' just because ', ' only because ', ' nevertheless ', ' specifically ', ' additionally ', ' not because ', ' for example ', ' in addition ', ' accordingly ', ' furthermore ', ' in response ', ' by contrast ', ' if and when ', ' in contrast ', ' nonetheless ', ' even though ', ' as a result ', ' separately ', ' in the end ', ' conversely ', ' as long as ', ' regardless ', ' ultimately ', ' even after ', ' as much as ', ' apparently ', ' presumably ', ' only after ', ' even still ', ' in return ', ' only when ', ' even then ', ' lest,once ', ' even when ', ' turns out ', ' as though ', ' similarly ', ' therefore ', ' typically ', ' meanwhile ', ' likewise ', ' moreover ', ' in short ', ' now that ', ' meantime ', ' although ', ' instead ', ' that is ', ' much as ', ' only if ', ' in fact ', ' even as ', ' neither ', ' further ', ' overall ', ' even if ', ' besides ', ' so that ', ' because ', ' in turn ', ' finally ', ' whereas ', ' thereby ', ' however ', ' indeed ', ' though ', ' unless ', ' rather ', ' in sum ', ' after ', ' until ', ' still ', ' while ', ' hence ', ' since ', ' first ', ' as if ', ' as it ', ' also ', ' when ', ' next ', ' thus ', ' plus ', ' then ', ' but ', ' yet ', ' nor ', ' for ', ' and ', ' if ', ' or ', ' as ', ' so '] # typical words used in a discoure relation 

def num_discourse(text: str) -> int:
    """
    Count the number of discourse relations in a comment. This is done by counting how many times discourse keywords appear in a comment. Feature 4 for thoughtful comment.

    Args:
        text (str): Comment by an user

    Returns:
        int: Number of discourse relations
    """

    text = remove_hashtag_mentions_urls(text)
    text = remove_emoji(text)
    text = replace_characters(text)

    count = 0
    for sentence in sent_tokenize(text):
        word_tokenizer = RegexpTokenizer(r'[-\'\w]+')
        tokenized_text = word_tokenizer.tokenize(sentence)
        tokenized_text = [w.lower() for w in tokenized_text]

        text_final = " ".join(tokenized_text)

        for ele in discourse_keywords:
            if ele in text_final:
                count += 1

    return count


# Thoughful comment feature 5
noun_tags = ['NN', 'NNS', 'NNP', 'NNPS'] # POS tags for nouns

def topic_doc_unigram(doc_words: list, k: int = 1) -> (defaultdict, list):
    """
    Creating unigram model for all the nouns in the policy topic text document. 

    Args:
        doc_words (list): list of tokenized words in the text document
        k (int): smoothing pseudo-count for each unigram

    Returns:
        [defaultdict]: unigram model with every unigram as key and its probabilty as value
        [list]: list of nouns in the document
    """
    doc_tags = nltk.pos_tag(doc_words)
    doc_nouns_tag = [tag[0].lower() for tag in doc_tags if tag[1] in noun_tags]
    doc_nouns = [w for w in doc_nouns_tag if re.search('^[a-z]+$', w)]

    num_words = len(doc_nouns)

    nouns_count = defaultdict(int)
    for word in doc_nouns:
        nouns_count[word] += 1
    
    vocab_size = len(nouns_count)

    doc_unigram = defaultdict(int)
    for key, v in nouns_count.items():
        doc_unigram[key] = (v + k) / (num_words + k*vocab_size)
    
    return doc_unigram, doc_nouns


def comment_unigram(comment: str, k: int = 1) -> (defaultdict, list):
    """
    Creating unigram model for all the nouns in an user's comment.

    Args:
        doc_words (list): list of tokenized words in the comment
        k (int): smoothing pseudo-count for each unigram

    Returns:
        [defaultdict]: unigram model with every unigram as key and its probabilty as value
        [list]: list of nouns in the comment
    """
    comment_words = word_tokenize(comment)
    comment_tags = nltk.pos_tag(comment_words)
    comment_nouns_tag = [tag[0].lower() for tag in comment_tags if tag[1] in noun_tags]
    comment_nouns = [w for w in comment_nouns_tag if re.search('^[a-z]+$', w)]

    num_words = len(comment_nouns)

    cmt_nouns_count = defaultdict(int)
    for word in comment_nouns:
        cmt_nouns_count[word] += 1
    
    vocab_size = len(cmt_nouns_count)

    cmt_unigram = defaultdict(int)
    for key, v in cmt_nouns_count.items():
        cmt_unigram[key] = (v + k) / (num_words + k*vocab_size)
    
    return cmt_unigram, comment_nouns


def KLDiv_relevance_score(doc_unigram: dict, comment_unigram: dict, doc_nouns: list, comment_nouns: list) -> int:
    """
    Calculate the KL-divergence relevance score between targetted document unigram model and comment unigram model. Feature 5 for thoughtful model. 

    Args:
        doc_unigram (dict): Targetted document unigram model with nouns only
        comment_unigram (dict): Comment unigram model with nouns only
        doc_nouns (list): list of nouns in targetted document
        comment_nouns (list): list of nouns in comment

    Returns:
        int: KL-divergence relevance score
    """
    total_vocab = set(doc_nouns).union(set(comment_nouns))

    kl_div = 0
    for word in total_vocab:
        if comment_unigram[word] * doc_unigram[word] != 0:
            kl_div += comment_unigram[word] * math.log(comment_unigram[word] / doc_unigram[word])
    
    return kl_div





