# -*- coding: utf-8 -*-
"""
Created on Tue May  1 19:57:28 2018

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
    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return text

data['clean_text'] = data['body_text'].apply(lambda x:clean_text(x))
data.head()

#Ngram veactorization

from sklearn.feature_extraction.text import CountVectorizer

ngram_vect = CountVectorizer(ngram_range=(2,2))
x_counts=ngram_vect.fit_transform(data['clean_text'])
print(x_counts.shape)
print(ngram_vect.get_feature_names())

data_sample = data [0:20]
ngram_vect_sample = CountVectorizer(ngram_range=(2,2))
x_counts_sample=ngram_vect_sample.fit_transform(data_sample['clean_text'])
print(x_counts_sample.shape)
print(ngram_vect_sample.get_feature_names())

x_df =pd.DataFrame(x_counts_sample.toarray())
x_df.columns=ngram_vect_sample.get_feature_names()
x_df