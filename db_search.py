import os
import numpy as np
import pickle
import heapq as hq
import sys
from datetime import datetime
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import db_config
import db_view


# setting checkpoint
db_config.config_setting_test()

# load inverted index and word list
with open(db_config.g_inverted_index, 'rb') as f:
    inv_idx, word_list = pickle.load(f)


def idf(k):
    try:
        return np.log10(db_config.g_total_docs / inv_idx[word_list.index(k)][1])
    except ValueError:
        return sys.float_info.epsilon  # avoid divide by zero


def w_kd(k, dp):
    # assuming dp is the tuple (doc#, freq) for k
    return idf(k) * dp[1]  # document weight = f_kd * idf_k


def w_kq(k, q):
    if len(q) < 4:  # definition of 'long' query varies; here we define as 4 or more words as long, for instance
        return idf(k)
    else:
        return idf(k) * q.count(k)  # if all words are unique in the query, the count factor has no effect


def convert_q_to_qry(q):
    qry = []
    for x in q:
        qry.append([x, w_kq(x, q)])

    return qry


def sort_by_w(l):
    for x in range(0, len(l)-1):
        for y in range(1, len(l)):
            if l[x][1] < l[y][1]:  # sort in descending order
                t = l[y]
                l[y] = l[x]
                l[x] = t


# prank() workspace and constant
docTbl = dict()
NAccum = db_config.g_total_docs * 0.1


def prank(qry, acc):
    # qry = [[k, w], [k, w], ...]
    # acc = [[Did, s], [Did, s], ...]
    # acc is initially empty; returned with most highly ranked
    sort_by_w(qry)
    # A* can have a small initial value -- will be updated anyway after first iteration
    a_star = db_config.g_prank_astar
    for q in qry:
        tau_ins = (db_config.g_prank_tau_ins * a_star) / (q[1] * idf(q[0]))
        tau_add = (db_config.g_prank_tau_add * a_star) / (q[1] * idf(q[0]))
        # handle case when the word does not exist
        try:
            fp = inv_idx[word_list.index(q[0])]
        except ValueError:
            continue  # if the word does not exist, continue on to the next one
        if fp[2] <= tau_add:  # if total frequency of this keyword is too small in corpus, ignore
            break
        new_score = fp[2] * idf(q[0]) * q[1]
        for dp in fp[3]:
            if (dp[0] in docTbl) or (w_kd(q[0], dp) > tau_ins):
                doc_nos = [x[1] for x in acc]  # get all document numbers in Acc as a list
                if dp[0] in doc_nos:
                    fnd = doc_nos.index(dp[0])  # find index of "this" document number
                    acc[fnd] = (acc[fnd][0] + new_score, acc[fnd][1])  # update corresponding score
                    new_score = acc[fnd][0]  # for comparison purpose below
                    acc = sorted(acc, reverse=True)
                else:
                    docTbl[dp[0]] = True
                    if len(acc) > NAccum:
                        popped = hq.heappushpop(acc, (new_score, dp[0]))
                        docTbl[popped[0]] = False
                    else:
                        hq.heappush(acc, (new_score, dp[0]))
            a_star = max(a_star, new_score)

    return np.array(acc)


def clean_and_stem_document(q, noise_words_path):
    # prepare noise words
    if noise_words_path == '':
        stop_words = set(stopwords.words('english'))
    else:
        with open(noise_words_path, 'rt') as f:
            stop_words = f.read()
        stop_words = word_tokenize(stop_words)

    # next, preprocess the new document by stemming and removing stopwords
    filtered_new_doc = []
    for w in q:
        if w not in stop_words:
            filtered_new_doc.append(db_config.g_stemmer.stem(w))

    return filtered_new_doc


def sort_by_score(v):
    return v[0]


def search(q):
    q = word_tokenize(q)
    q = clean_and_stem_document(q, db_config.g_noise_words_path)
    qry = convert_q_to_qry(q)
    acc = prank(qry, [])
    return acc


def main():
    # # Debug
    #
    # # Short Query test
    # acc = search('probability machine learning')
    # acc = search('support vector machine VC')
    # db_view.show_result_ait_to_file(acc)
    #
    # time.sleep(1)  # without this, previous result will probably be overwritten

    # # Long Query test
    # acc = search('probability machine learning model intelligent sensor network')
    # db_view.show_result_ait_to_file(acc)
    #
    # # Test other functions (this is for part 4)
    # print(db_view.get_result_ait_at_k(acc, 10))
    # print(db_view.get_result_ait_range(acc, 5, 10))
    # print(db_view.get_result_ait_top_n(acc, 3))
    pass


main()
