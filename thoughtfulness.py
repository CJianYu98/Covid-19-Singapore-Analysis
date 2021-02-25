import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from unigram import *
from preprocess import *

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
def get_sentence_length(text: str) -> int:
    text = replace_characters(text)
    text = remove_hashtag_mentions_urls(text)
    text = remove_emoji(text)

    word_tokenizer = RegexpTokenizer(r'[-\'\w]+')
    tokenized_text = word_tokenizer.tokenize(text)
    return len(tokenized_text)

# Thoughtful comment feature 2
def comment_likelihood(text: str, train_model: UnigramModel) -> float:
    text_replaced = replace_characters(text)
    txt = ''

    for tokenized_sentence in generate_tokenized_sentences(text_replaced):
        sent = ','.join(tokenized_sentence)
        txt += sent

    text_counter = UnigramCounter(txt)

    text_avg_log_likelihood = train_model.evaluate(text_counter)
    return text_avg_log_likelihood


def news_articles_unigram(file_name: str) -> UnigramModel:
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
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
def get_num_verbs(text: str) -> int:

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
discourse_keywords = [' what’s more as a matter of fact ', ' at least partly because ', ' on the one hand indeed ', ' in large part because ', ' particularly because ', ' largely as a result ', ' particularly since ', ' especially because ', ' on the other hand ', ' primarily because ', ' at the same time ', ' especially since ', ' not only because ', ' reportedly after ', ' on the contrary ', ' in part because ', ' largely because ', ' particularly as ', ' perhaps because ', ' particularly if ', ' on the one hand ', ' simply because ', ' merely because ', ' in other words ', ' partly because ', ' mainly because ', ' in particular ', ' for one thing ', ' as if besides ', ' one day after ', ' especially as ', ' by comparison ', ' especially if ', ' also although ', ' consequently ', ' on the whole ', ' when if only ', ' particularly ', ' incidentally ', ' just because ', ' only because ', ' nevertheless ', ' specifically ', ' additionally ', ' not because ', ' for example ', ' in addition ', ' accordingly ', ' furthermore ', ' in response ', ' by contrast ', ' if and when ', ' in contrast ', ' nonetheless ', ' even though ', ' as a result ', ' separately ', ' in the end ', ' conversely ', ' as long as ', ' regardless ', ' ultimately ', ' even after ', ' as much as ', ' apparently ', ' presumably ', ' only after ', ' even still ', ' in return ', ' only when ', ' even then ', ' lest,once ', ' even when ', ' turns out ', ' as though ', ' similarly ', ' therefore ', ' typically ', ' meanwhile ', ' likewise ', ' moreover ', ' in short ', ' now that ', ' meantime ', ' although ', ' instead ', ' that is ', ' much as ', ' only if ', ' in fact ', ' even as ', ' neither ', ' further ', ' overall ', ' even if ', ' besides ', ' so that ', ' because ', ' in turn ', ' finally ', ' whereas ', ' thereby ', ' however ', ' indeed ', ' though ', ' unless ', ' rather ', ' in sum ', ' after ', ' until ', ' still ', ' while ', ' hence ', ' since ', ' first ', ' as if ', ' as it ', ' also ', ' when ', ' next ', ' thus ', ' plus ', ' then ', ' but ', ' yet ', ' nor ', ' for ', ' and ', ' if ', ' or ', ' as ', ' so ']




