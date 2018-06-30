# -*- coding: utf-8 -*-
"""
Created on Sat May  5 10:15:12 2018

@author: jayesh
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

def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100
data['body_len'] = data['body_text'].apply(lambda x:len(x) - x.count(" "))
data['punt%'] = data['body_text'].apply(lambda x:count_punct(x))


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

#feature
x_feature = pd.concat([data['body_len'],data['punt%'],pd.DataFrame(x_tfid.toarray())],axis=1)
x_feature.head()

#data_sample = data[0:20]
#Tfid_vect_sample = TfidfVectorizer(analyzer=clean_text)
#x_tfid_sample = Tfid_vect_sample.fit_transform(data_sample['body_text'])
#print(x_tfid_sample.shape)
#print(Tfid_vect_sample.get_feature_names())
#
#x_df =pd.DataFrame(x_tfid_sample.toarray())
#x_df.columns=Tfid_vect_sample.get_feature_names()
#x_df

from sklearn.ensemble import RandomForestClassifier

print(RandomForestClassifier())

from sklearn.model_selection import KFold,cross_val_score

rf = RandomForestClassifier(n_jobs=-1)
k_fold = KFold(n_splits = 5)
cross_val_score(rf,x_feature,data['label'],cv = k_fold,scoring='accuracy',n_jobs=-1)