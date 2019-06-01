# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 00:05:25 2019

@author: Anirban
"""

from .tasks.task1_readdata import read_data
from .tasks.task2_processdata import process_file
from .tasks.task3_createfeatures import create_features
from .tasks.task4_modeltopic import model_topic

def run_wf(source, target, file):
    # read data
    df = read_data(source, target, file)
    # process data
    df = process_file(df, target, (file+'.pkl'))
    # create feature
    _create_features = create_features(df.processed,df.con)
    # topic model
    _model_topic = model_topic(df)
    
    
    