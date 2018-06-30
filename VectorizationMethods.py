# -*- coding: utf-8 -*-
"""
Created on Tue May  1 19:08:32 2018

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

#Applay count vectorization

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(analyzer=clean_text)
x_counts=count_vector.fit_transform(data['body_text'])
print(x_counts.shape)
print(count_vector.get_feature_names())


#select first 20 
data_sample = data [0:20]
count_vect_sample = CountVectorizer(analyzer=clean_text)
x_counts_sample =count_vect_sample.fit_transform(data_sample['body_text'])
print(x_counts_sample.shape)
print(count_vect_sample.get_feature_names())
x_df =pd.DataFrame(x_counts_sample.toarray())
x_df.columns=count_vect_sample.get_feature_names()
