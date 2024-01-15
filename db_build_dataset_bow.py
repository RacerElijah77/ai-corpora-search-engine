# Some codes following a tutorial available on:
#   https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
from itertools import filterfalse
import pickle
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer

import db_config

# More efficient method to fit corpus dataset into model
def generate_dataset_file(corpora, descriptor_label, dataset_path):

    with open(corpora, 'rb') as f:
        all_docs, _, _ = pickle.load(f)

    with open(descriptor_label, 'rb') as f:
        all_labels, all_cats_short = pickle.load(f)

    y = all_labels

    # takes original sentences as is
    x = []

    for d in all_docs:
        x.append(d[3] + ' ' + d[10])

    print("Start computing tf-idf features")
    t0 = time()
    vect = TfidfVectorizer(max_df=db_config.g_bow_max_df, min_df= db_config.g_bow_min_df)
    x = vect.fit_transform(x)
    duration_time = time() - t0

    print(f'Done is {duration_time:.3f} seconds')

    with open(dataset_path, 'wb') as f:
        pickle.dump([x,y,vect], f)

def get_feature(dataset_path, new_doc):
    with open(dataset_path, 'rb') as f:
        _, _, vect = pickle.load(f)

    s = ' '
    filtered_new_doc = [s.join(new_doc)]

    t0 = time()
    x = vect.transform(filtered_new_doc)
    duration_test = time() - t0
    print(f'Done is {duration_test:.3f} seconds')

    return x

def main():
    if not os.path.exists(db_config.g_dataset_path_bow):
        generate_dataset_file(db_config.g_corpora_file, db_config.g_descriptor_label, db_config.g_dataset_path_bow)

    # q = ['probability', 'machine', 'learning']
    # z = get_feature(db_config.g_dataset_path_bow, q)
    # print("Hello" + str(z))


main()
