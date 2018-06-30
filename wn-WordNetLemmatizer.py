# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 18:30:26 2018

@author: Jayesh
"""

import nltk
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()

dir(wn)
print(wn.lemmatize('goose'))
print(wn.lemmatize('geese'))

import pandas as pd
import string
import re

pd.set_option("display.max_colwidth",100)
stopwords = nltk.corpus.stopwords.words('english')
data = pd.read_csv("SMSSpamCollection.tsv", sep ='\t')
data.columns =['label','body_text']

data.head()

def clean_text(text):
    text="".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = [word for word in tokens if word not in stopwords]
    return text
data['body_text_nonstop']= data['body_text'].apply(lambda x:clean_text(x.lower()))
data.head()  

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text
data['body_text_lemmatized'] = data['body_text_nonstop'].apply(lambda x:lemmatizing(x))
data.head()  