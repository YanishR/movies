##################################################################
# Movie Review Sentiment Analysis
# Members: Amber, Basel, Suvinay, Yanish
#

#import numpy

import re
reviews_train = []
for line in open('./aclImdb/movie_data/full_train.txt', 'r'):
  reviews_train.append(line.strip())

reviews_test = []
for line in open('./aclImdb/movie_data/full_test.txt', 'r'):
  reviews_test.append(line.strip())
