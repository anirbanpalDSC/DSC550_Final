# -*- coding: utf-8 -*-
"""
Created on Fri May 31 23:52:13 2019

@author: Anirban
"""

# standard library imports
import nltk
import numpy as np
import re

# third party imports
from gensim import corpora
from gensim.models import LdaModel

def model_topic(df):
    """
    Create the topic modeling based on the corpus
    Args:
        param1(dataframe): dataframe with corpus
    Returns:
        create topic model and writes in the report markdown
    """
    
    df['tokenized'] = np.nan
    for row in range(len(df)):
        df['tokenized'].iloc[row] = nltk.word_tokenize(df['processed'].iloc[row])
    dictionary = corpora.Dictionary(df.Tokenized)
    # create bag of words format
    bow = [dictionary.doc2bow(text) for text in df.Tokenized]
    lda = LdaModel(bow, num_topics=10, id2word = dictionary, passes=20)
    topics = lda.print_topics(num_topics=10, num_words=1)
    with open('C:/Users/Anirban/Desktop/Masters/MSDSC/DSC550/Excercise/DSC550_Final/reports/report_task4.md', 'a+') as mdWriter:
        for i in range(len(df)):
            line_extract = df['tokenized'].iloc[i]
            new_review_bow = dictionary.doc2bow(line_extract)
            new_review_lda = lda[new_review_bow]
            topic_no = new_review_lda[0][0]
            topic = re.findall('[A-z]\w+', topics[topic_no][1])[0]
            mdWriter.writelines(('Comment:', str(i+1),' Topic:', str(topic),'\n'))
