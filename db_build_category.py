import pickle
import os
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt  # for debugging

import db_config


def compute_descriptor_labels(simple_binary_copora, descriptor_label_path):
    # open simple corpora
    with open(simple_binary_copora, 'rb') as f:
        all_docs, _ = pickle.load(f)

    # collect all descriptors
    all_cats = []  # should end up with 374 categories, initially
    for one_doc in all_docs:
        all_cats = list(set(all_cats + list(set(one_doc[4]))))

    print(all_cats)
    # get the histogram count of categories to drop negligible categories
    all_cats_cnt = np.zeros(len(all_cats), dtype='int')
    for one_doc in all_docs:
        for one_cat_in_doc in one_doc[4]:
            # no need to check existence since at least one item should match
            loc = all_cats.index(one_cat_in_doc)
            all_cats_cnt[loc] = all_cats_cnt[loc] + 1

    # let's take categories with at least 50 documents (should be 21)
    # itertools.compress(x, y) is a very handy function to find elements in the list x
    # where corresponding y value is True. For example, x = ['C', 'C++', 'Java'] and
    # y = [True, False, True], then it returns ['C', 'Java']. Note that one should use
    # list() to make it a list, or use other ways (e.g., for..in structure) to iterate
    # through the returned object.
    cat_loc = all_cats_cnt >= 50
    all_cats_short = list(compress(all_cats, cat_loc))
    all_cats_cnt_short = list(compress(all_cats_cnt, cat_loc))

    # make sure these 64 categories cover all documents (IMPORTANT!)
    # if there are documents that fall in this condition, we assign a new label [UNKNOWN]
    for ii in range(0, len(all_docs)):
        a = set(all_docs[ii][4])
        b = set(all_cats_short)
        if len(a.intersection(b)) == 0:
            all_docs[ii][4] = ['UNKNOWN']

    # add category
    all_cats_short.append('UNKNOWN')  # now 22 categories

    counter = 0
    for x in all_cats_short:
        print(str(counter) + " " + x)
        counter += 1

    # double check
    for one_doc in all_docs:
        a = set(one_doc[4])
        b = set(all_cats_short)
        if len(a.intersection(b)) == 0:
            print('Should not reach here!')

    # now we encode and assign labels; we basically use index for encoding
    all_labels = np.zeros(len(all_docs), dtype='int')
    for ii in range(0, len(all_docs)):
        min_cat_loc = 0
        min_cat_cnt = 99999  # Python3 integer is unbounded...

        # find category with **min** document frequency
        # WHY? because categories that rarely show up probably closely relevant to this particular document
        #      note that we already eliminated trivial ones -- thus it's highly likely this one's important!
        one_doc = all_docs[ii]
        for one_cat_in_doc in one_doc[4]:
            if (one_cat_in_doc != 'UNKNOWN') and (one_cat_in_doc in all_cats_short):
                one_loc = all_cats_short.index(one_cat_in_doc)
                if all_cats_cnt_short[one_loc] < min_cat_cnt:
                    min_cat_loc = one_loc
                    min_cat_cnt = all_cats_cnt_short[min_cat_loc]

        # If not found
        if min_cat_cnt == 99999:
            all_labels[ii] = len(all_cats_short) - 1  # UNKNOWN category
        else:
            all_labels[ii] = min_cat_loc

    # all docs would represent through the min label [0-21], need this for part 2?
    print(len(all_labels))

    # DEBUG: if you want to see the result...
    #        note that there's always some class(es) not selected at all; guess why is this happening
    # plt.hist(all_labels, bins=len(all_cats_short)+1)
    # plt.show()

    # write binary
    with open(descriptor_label_path, 'wb') as f:
        pickle.dump([all_labels, all_cats_short], f)


def main():
    if not os.path.exists(db_config.g_descriptor_label):
        compute_descriptor_labels(db_config.g_corpora_simple, db_config.g_descriptor_label)

main()
