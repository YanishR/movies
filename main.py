import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from matplotlib import pyplot as plt
import string
import itertools
import glob
import os
import re
#Import helper methods
from mt import *
#This code is for reading the multiple files in one folder into a list
# read neg files in train folder
data_path = "./aclImdb/movie_data/"

train = extract_data(data_path + "full_train.txt")
test = extract_data(data_path + "full_test.txt")

cv = CountVectorizer(binary=True)
cv.fit(train)
X = cv.transform(train)
X_test = cv.transform(test)

target = [1 if i < len(train)/2 else 0 for i in range(len(train))]

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)

print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(X_test)))

feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1],
    reverse=True)[:5]:
    print (best_positive)

for best_negative in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1])[:5]:
    print (best_negative)
