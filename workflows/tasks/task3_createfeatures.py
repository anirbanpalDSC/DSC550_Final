# -*- coding: utf-8 -*-
"""
Created on Fri May 31 23:32:33 2019

@author: Anirban
"""

# standard library imports
import datetime

# third party imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def create_features(features, target):
    """
    
    """
    
    cv = CountVectorizer()
    cv_features = cv.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(cv_features, target, test_size=0.25, random_state=101)
    try:
        lr2 = joblib.load('C:/Users/Anirban/Desktop/Masters/MSDSC/DSC550/Excercise/DSC550_Final/models/model.pkl')
    except:
        lr2 = LogisticRegression(penalty='l2')
    mod_lr2 = lr2.fit(X_train, y_train)
    y_pred = mod_lr2.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    joblib.dump(mod_lr2, 'C:/Users/Anirban/Desktop/Masters/MSDSC/DSC550/Excercise/DSC550_Final/models/model.pkl')
    with open('C:/Users/Anirban/Desktop/Masters/MSDSC/DSC550/Excercise/DSC550_Final/reports/report_task3.md', 'a+') as mdWriter:
        mdWriter.writelines(('Logistic Regression using penalty lr2', '\n',
                             'runtime:', str(datetime.datetime.now()),'\n',
                             'Model accuracy:', str(accuracy_test),'\n'))