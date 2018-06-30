# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 20:18:47 2018

@author: Admin
"""

#understanding casting

from sklearn import random_projection
import numpy as np

rng = np.random.RandomState(0)
X = rng.rand(0,2000)
X = np.array(X,dtype = 'float32')
X.dtype

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit(X)
X_transform = transformer.transform(X_new)
X_transform.dtype
