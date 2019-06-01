# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:29:14 2019

@author: Anirban
"""

# standard library imports
import pandas as pd
import json

# third party imports
from pandas.io.json import json_normalize


def read_data(source, target, file):
    """
    Read all source files in a single dataframe
    Args:
        param1(str): source file location
        param2(str): target file location
        param3(str): file name (used for both source and target)
    Returns:
        dataframe with source data
    """

    data = []
    with open((source + file)) as f:
        for line in f:
            data.append(json.loads(line))        
    df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
    
    return df