# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 12:06:07 2018

@author: Admin
"""

import tensorflow as tf
import pandas as pd
import numpy as np
hello = tf.constant('Hello, TensorFlow!')

g = tf.Graph()
with g.as_default():
  # Assemble a graph consisting of the following three operations:
  #   * Two tf.constant operations to create the operands.
  #   * One tf.add operation to add the two operands.
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  my_sum = tf.add(x, y, name="x_y_sum")


  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
    print (my_sum.eval())