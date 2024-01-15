import os
from nltk.stem import SnowballStemmer


# corpus specific options
g_stemmer = SnowballStemmer('english')
g_xml_file_path = './AIT/'  # change this to work in your setting
g_db_file_path = './database/files.txt'
g_corpora_file = './database/corpora.pickle'
g_noise_words_path = './AIT/stop.wrd'  # change this to work in your setting
g_corpora_simple = './database/corpora_simple.pickle'
g_inverted_index = './database/inverted_index.pickle'
g_total_docs = 2361 + 1691 + 505  # you can calculate this from inv_dic as well (only counting AIT corpus!)
g_min_qry_threshold = 4  # query needs to have at least this many words to consider TF

# classification model specific options
g_min_threshold_for_category = 50  # at least this many documents belong to a category
g_descriptor_label = './database/descriptor_label.pickle'
g_dataset_path_tfidf = './database/dataset_tfidf.pickle'
g_dataset_path_bow = './database/dataset_bow.pickle'

# DecisionTree options
g_trained_model_path_tfidf_dtree = './database/trained_model_tfidf_dtree.pickle'
g_dtree_max_depth = 500
g_dtree_rand_states = 42

# BoW options
g_trained_model_path_bow_dtree = './database/trained_model_bow_dtree.pickle'
g_bow_max_df = 0.5
g_bow_min_df = 5

# Word2Vec options
g_trained_model_path_word2vec_nn = './database/trained_model_word2vec_nn.keras'
g_max_features = 10000
g_sequence_length = 250
g_embedding_dim = 16
g_epochs = 10

# classification model choice (should be one of the following)
#   g_trained_model_path_tfidf_dtree
#   g_trained_model_path_bow_dtree
#   g_trained_model_path_word2vec_dtree
g_trained_model = g_trained_model_path_tfidf_dtree

# search engine option
g_config_setting_test_passed = False
g_prank_astar = 0.1
g_prank_tau_ins = 0.07
g_prank_tau_add = 0.01


def config_setting_test():
    global g_config_setting_test_passed
    if not g_config_setting_test_passed:
        # Need to run the following first to pass the next two lines:
        #   db_build_corpora_and_inverted_index.py
        assert os.path.exists(g_corpora_simple), 'Error: Corpora must be build first.'
        assert os.path.exists(g_inverted_index), 'Error: Inverted index must be build first.'
        # Need to run the following first to pass the next line:
        #   db_build_category.py
        assert os.path.exists(g_descriptor_label), 'Error: Category label file missing.'
        # Need to run the following first to pass the next line:
        #   db_build_dataset_*.py (e.g., db_build_dataset_tfidf.py)
        #   db_build_classifier_*.py (e.g., db_build_classifier_dtree.py)
        assert os.path.exists(g_trained_model), 'Error: Classification model missing.'

        print(os.path.basename(__file__) + ': Passed configuration test.')
        g_config_setting_test_passed = True

