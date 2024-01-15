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

clf = DecisionTreeClassifier(max_depth = db_config.g_dtree_max_depth, random_state= db_config.g_dtree_rand_states)
def select_clf_algorithm(value, deleteValue):

    # KNN Classifier algorithm
    global clf
    if(value == '1'):
        clf = KNeighborsClassifier(n_neighbors=int(math.sqrt(21)))
        print(clf)
    # Random Forest algorithm
    elif(value == '2'):
        clf = RandomForestClassifier(max_depth = db_config.g_dtree_max_depth, random_state= db_config.g_dtree_rand_states)
        print(clf)
    elif(value == '0'):
        clf = DecisionTreeClassifier(max_depth = db_config.g_dtree_max_depth, random_state= db_config.g_dtree_rand_states)
        print(clf)

    # If the user wants to choose a new classifier algorithm , they must type 9999 in the second input box
    if(deleteValue == '9999'):
        os.remove(db_config.g_trained_model_path_bow_dtree)
        fit_model(db_config.g_dataset_path_bow, db_config.g_trained_model_path_bow_dtree)

def fit_model(dataset_path, trained_model_path):
    with open(dataset_path, 'rb') as f:
        x, y, _ = pickle.load(f)

    with open(db_config.g_descriptor_label, 'rb') as g:
        _, all_cats_short = pickle.load(g)

    # print("X Vector" + str(x))
    # print("Y Vector" + str(len(y)))
    # for j in y:
    #     print(str(j) + "\t" + str(all_cats_short[j]))

    print("Training Classifier using :" + str(clf))
    clf.fit(x, y)

    with open(trained_model_path, 'wb') as f:
        pickle.dump(clf, f)

    print("Cross validation analysis for " + str(clf))
    score = cross_val_score(clf, x,y, cv=5)
    print("%0.3f accuracy with a std-dev of %0.3f" % (score.mean(), score.std()))

    print("Done!!")

def predict_with_model(trained_model_path, z):
    with open(trained_model_path, 'rb') as f:
        a = pickle.load(f)

    return a.predict(z.reshape(1,-1))

# Part1 for HW (Interface funtion for web_server.py)
def classify_query(qry):
    # z = 'philosophy with some random text machine learning'
    z = (word_tokenize(qry)) # NLTK library
    print("Original query: " + str(z))

    # obtain feature vector from query z
    z_tfidf = db_build_dataset_bow.get_feature(db_config.g_dataset_path_bow, z)
    # print(z_tfidf)
    # print(len(z_tfidf))

    pred_label_tfidf = predict_with_model(db_config.g_trained_model_path_bow_dtree, z_tfidf)
    print(pred_label_tfidf)

    # obtain list of top 20 descriptors (classes)
    with open(db_config.g_descriptor_label, 'rb') as g:
        _, all_cats_short = pickle.load(g)

    # printing the top classifiers for checking
    # counter = 0
    # for f in all_cats_short:
    #     print(str(counter) + " " + str(f))
    #     counter += 1

    # this would probably be the label that will be seen to web_server.py
    print(all_cats_short[int(pred_label_tfidf)])

    return all_cats_short[int(pred_label_tfidf)]

def main():
    if not os.path.exists(db_config.g_trained_model_path_bow_dtree):
        fit_model(db_config.g_dataset_path_bow, db_config.g_trained_model_path_bow_dtree)

    # classify_query("Business administration")

main()
