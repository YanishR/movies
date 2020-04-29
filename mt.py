import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from matplotlib import pyplot as plt
import string
import itertools
import glob
import os
import re

REVIEWS=2
def extract_data(file_path):

    # Clean data with regex
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    # return list
    data = []

    for line in open(file_path, 'r'):
        review = REPLACE_NO_SPACE.sub("",line.strip().lower())
        data.append(REPLACE_WITH_SPACE.sub(" ", review))
    
    return data #[:REVIEWS]

