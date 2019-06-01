# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:40:30 2019

@author: Anirban
"""

# standard library imports
import string

# third party imports
from nltk.corpus import stopwords
from nltk import PorterStemmer


def process_text(text):
    """
    Takes a text block, removes punctuation, stop words, commoner 
    morphological and inflexional endings from words in English
    Args:
        param1(str): text block to be cleaned
    Returns:
        processed text
    """
 
    #set stop words
    stop_words = set(stopwords.words('english'))
    # remove punctuation
    processedtxt = [ch for ch in text if ch not in string.punctuation]
    processedtxt = ''.join(processedtxt)
    # lower case
    processedtxt = processedtxt.lower()
    # remove stop words
    processedtxt = [w for w in processedtxt.split() if w not in stop_words]
    # create stemmer object
    ps = PorterStemmer()
    # remove metaphors
    processedtxt = [ps.stem(word) for word in processedtxt]
    processedtxt = ' '.join(processedtxt)
    
    return processedtxt


def process_file(df, target, file):
    """
    Apply text processing and stemming to a dataframe
    Args:
        param1(df): input dataframe
        param2(str): target location to store pickled file of processed data
        file(str): 
    """
    
    df['processed'] = df['txt'].apply(process_text)
    # save processed file
    df.to_pickle((target + file))
    
    return df