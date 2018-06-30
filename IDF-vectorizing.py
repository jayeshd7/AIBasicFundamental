# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:46:23 2018

@author: Jayesh
"""

import pandas as pd
import nltk
import re
import string

pd.set_option("display.max_colwidth",100)
stopwords = nltk.corpus.stopwords.words("english")
ps = nltk.PorterStemmer()
data = pd.read_csv("SMSSpamCollection.tsv",sep ='\t')
data.columns = ['label','body_text']

def clean_text(text):
    text="".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
data['body_text_nonstop']= data['body_text'].apply(lambda x:clean_text(x.lower()))
data.head()  


from sklearn.feature_extraction.text import TfidfVectorizer

Tfid_vect = TfidfVectorizer(analyzer=clean_text)
x_tfid = Tfid_vect.fit_transform(data['body_text'])
print(x_tfid.shape)
print(Tfid_vect.get_feature_names())

data_sample = data[0:20]
Tfid_vect_sample = TfidfVectorizer(analyzer=clean_text)
x_tfid_sample = Tfid_vect_sample.fit_transform(data_sample['body_text'])
print(x_tfid_sample.shape)
print(Tfid_vect_sample.get_feature_names())

x_df =pd.DataFrame(x_tfid_sample.toarray())
x_df.columns=Tfid_vect_sample.get_feature_names()
x_df