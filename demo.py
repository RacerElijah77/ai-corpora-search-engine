import os
import pickle
import math
import numpy as np
from nltk.tokenize import word_tokenize

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import db_config
import db_build_dataset_bow

with open(db_config.g_dataset_path_bow , 'rb') as f:
    x, y, vect = pickle.load(f)

print(x)

print(y)

print(vect)

with open(db_config.g_corpora_file, 'rb') as fc:
    all_docs, _, _ = pickle.load(fc)

ctr = 0
for i in all_docs:
    print("j" + str(i[3]))
    ctr += 1

print("Hello" + str(ctr))
