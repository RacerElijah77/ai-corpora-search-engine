import os
import pickle
import numpy as np
import math
from datetime import datetime

import db_config


# setting checkpoint
db_config.config_setting_test()


def show_result_ait_to_file(acc):
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open('result_' + date_time + '.txt', 'wt') as fr:

        with open(db_config.g_corpora_file, 'rb') as fc:
            all_docs, _, _ = pickle.load(fc)

        for r in acc:
            print('#####################', file=fr)
            print('* Score: ' + r[0], file=fr)
            print('* Doc #: ' + r[1], file=fr)
            print('* Title: ' + str(np.array(all_docs[int(r[1])-1][3])), file=fr)
            print('* Autho: ' + str(np.array(all_docs[int(r[1])-1][4])), file=fr)
            print('* Desc : ' + str(np.array(all_docs[int(r[1])-1][7])), file=fr)
            print('* Abstr: ' + str(np.array(all_docs[int(r[1])-1][10])), file=fr)

    del all_docs  # save some memory


def get_result_ait_at_k(acc, k):
    # return K-th rank document at full detail
    with open(db_config.g_corpora_file, 'rb') as fc:
        all_docs, _, _ = pickle.load(fc)

    # part 2 of the HW
    with open(db_config.g_dataset_path_bow, 'rb') as f:
        x, y, vect = pickle.load(f)

    # if y[0] = 7 -> first document with class label of 7
    # if y[1] = 2 -> second document with class label of 2

    # Obtain the file of the top 21 classifiers
    with open(db_config.g_descriptor_label, 'rb') as g:
        _, all_cats_short = pickle.load(g)

    # part 2 of the HW
    for g in range(0,len(all_docs)):
        class_string = all_cats_short[int(y[g])]
        all_docs[g].insert(0, class_string)
        # print("Modified" + str(all_docs[i]))

    # perform deep copy (class_label, score, doc#, title, description (label), abstract)
    res = [str(np.array(all_docs[int(acc[k][1]) - 1][0])),
           acc[k][0],
           acc[k][1],
           str(np.array(all_docs[int(acc[k][1])-1][4])),
           str(np.array(all_docs[int(acc[k][1])-1][5])),
           str(np.array(all_docs[int(acc[k][1])-1][8])),
           str(np.array(all_docs[int(acc[k][1])-1][11]))]

    del all_docs  # save some memory
    return res

def get_result_ait_range(acc, n_beg, n_end):
    # return document with ranking [n_beg, n_end) in detail
    res = []

    with open(db_config.g_corpora_file, 'rb') as fc:
        all_docs, _, _ = pickle.load(fc)

    if len(acc) < n_beg:
        n_beg = 0
        n_end = len(acc)
    elif len(acc) < n_end:
        n_end = len(acc)

    for d in range(n_beg, n_end):
        res.append(get_result_ait_at_k(acc, d))

    del all_docs  # save some memory
    return res


# could use this function for part 4?
def get_result_ait_top_n(acc, n):
    # return top N ranked documents in detail
    return get_result_ait_range(acc, 0, n)


def main():
    # test code is in db_search.py
    pass
