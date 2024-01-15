from itertools import filterfalse
import pickle
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import db_config


## Code for making vector feature representation first (Matrix)

def generate_dataset_file(inverted_index, descriptor_label, dataset_path):
    ndoc = db_config.g_total_docs

    # first, load inverted index and labels
    with open(inverted_index, 'rb') as f:
        inv_dic, word_list = pickle.load(f)

    # 20 labels are stored (alongside binary presence in all_cats_short)
    with open(descriptor_label, 'rb') as f:
        all_labels, all_cats_short = pickle.load(f)

    # response vector Y is easy. we already have it as all_labels, period
    y = all_labels

    # now we need to build input feature matrix X
    print('Start computing TF-IDF features...')
    nkw = len(word_list)
    # make it half precision to get the result faster
    x = np.zeros([ndoc, nkw], dtype='float16')
    for ii in range(0, ndoc):
        for jj in range(0, nkw):
            # Note that, since we are traversing ALL keywords anyway, we don't need .index()
            e = inv_dic[jj]
            for tf in filterfalse(lambda z: int(z[0]) != ii, e[3]):
                x[ii, jj] = np.sum(tf[1] * np.log10(ndoc / e[1]))

        if ii % 100 == 0:
            print('...done ', ii, '/', ndoc)

    print('ALL DONE!')

    # write binary
    with open(dataset_path, 'wb') as f:
        pickle.dump([x, y, word_list], f)


def get_feature(inverted_index, noise_words_path, new_doc):
    ndoc = db_config.g_total_docs

    # NOTE: if the new_doc is already stemmed + removed noise words,
    # repeated stemming/noise removal should have no effect!

    # first, load inverted index. but we don't need labels
    with open(inverted_index, 'rb') as f:
        inv_dic, word_list = pickle.load(f)

    # prepare noise words
    if noise_words_path == '':
        stop_words = set(stopwords.words('english'))
    else:
        with open(noise_words_path, 'rt') as f:
            stop_words = f.read()
        stop_words = word_tokenize(stop_words)

    # next, preprocess the new document by stemming and removing stopwords
    filtered_new_doc = []
    for w in new_doc:
        if w not in stop_words:
            filtered_new_doc.append(db_config.g_stemmer.stem(w))

    # print("FILTERED QUERTY: " + str(filtered_new_doc))

    # compute tf-idf for each term
    qry = []
    for k in filtered_new_doc:
        try:
            if len(filtered_new_doc) < 4:
                qry.append([k, np.log10(ndoc / inv_dic[word_list.index(k)][1])])
            else:
                qry.append([k, np.log10(ndoc / inv_dic[word_list.index(k)][1]) * filtered_new_doc.count(k)])
        except ValueError:
            qry.append([k, 0.0])

    # now we need to build input feature vector x
    nkw = len(word_list)
    # make it half precision to get the result faster
    x = np.zeros(nkw, dtype='float16')
    for w in qry:
        try:
            # it is possible that feature vector x may contain a word not in our word list
            x[word_list.index(w[0])] = w[1]
        except ValueError:
            continue

    # print("Input feature vector x: " + str(x))
    # print("Length of feature vector x: " + str(len(x)))
    return x


def main():
    # We need to build features. We can build a feature something like TF-IDF
    if not os.path.exists(db_config.g_dataset_path_tfidf):
        generate_dataset_file(db_config.g_inverted_index, db_config.g_descriptor_label, db_config.g_dataset_path_tfidf)

    # # utility function usage to compute the tf-idf feature for an arbitrary new (query) document
    q = ['probability', 'machine', 'learning', 'business']
    z = get_feature(db_config.g_inverted_index, db_config.g_noise_words_path, q)

    # # debug: should return the same number as length of the query
    print(np.count_nonzero(z))
    print(len(z))




main()
