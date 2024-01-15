import glob
import os
import time
from stat import *
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import db_config


def create_files_database_from_path(xml_file_path, db_file_path):
    f = open(db_file_path, 'w')
    # unlike documents, files do not have pre-assigned IDs. we need to assign them
    fid = 0
    # reinitialize the total documents
    db_config.g_total_docs = 0
    # for each XML file in the provided corpus path...
    for a_file in glob.glob(xml_file_path + '*.xml'):
        print('Found: ' + a_file)
        # get basic information about the file
        st = os.stat(a_file)
        a_file_date = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(st[ST_MTIME]))
        print(' Time: ' + str(a_file_date))
        a_file_size = st[ST_SIZE]
        print(' Size: ' + str(a_file_size))
        # open and read whole content from the file, and count number of documents
        with open(a_file) as a_open_file:
            content = a_open_file.read()
        a_file_ndocs = content.count("<THESIS>")
        print(' NDoc: ' + str(a_file_ndocs))
        f.write(str(fid) + ' "' + a_file + '" "' + str(a_file_date) + '" '
                + str(a_file_size) + ' ' + str(a_file_ndocs) + '\n')
        fid = fid + 1
        # update global variable
        db_config.g_total_docs += a_file_ndocs
    f.close()


def line_char_counting_reader(f, prev_line, prev_char):
    # this function is a helper function to keep track of running count of line number of characters
    s = f.readline()
    new_line = prev_line + 1
    new_char = prev_char + len(s)
    return s, new_line, new_char


def parse_ait_corpora(file_db_path, corpora_file):
    # we only deal with the corpus files we identified in create_files_database_from_path()
    files = []
    with open(file_db_path) as f:
        for l in f:
            files.append(l.split())

    print('Parsing XML files...')

    all_docs = []  # this list collects all "document" content we defined (title, abstract, etc.)
    # for each file...
    for file in files:
        f = open(file[1].replace('"', ''))
        line_cnt = 0
        char_cnt = 0
        beg_pos = 0
        txt_pos = 0
        end_pos = 0
        num_lines_beg = 0
        num_lines_abs = 0
        while True:
            # read a line from file, and see if this starts a new document (<THESIS>)
            a_line, line_cnt, char_cnt = line_char_counting_reader(f, line_cnt, char_cnt)
            if a_line.find('<THESIS>') >= 0:
                beg_pos = char_cnt - len(a_line)
                num_lines_beg = line_cnt - 1
                one_doc = [file[0]]  # NOTE: much better data structure choice would be Python's dictionary
                # mechanically parse each tag within the document
                while True:
                    a_line, line_cnt, char_cnt = line_char_counting_reader(f, line_cnt, char_cnt)
                    if a_line.find('</THESIS>') >= 0:
                        end_pos = char_cnt
                        num_lines_tot = line_cnt - num_lines_beg
                        one_doc.append([beg_pos, txt_pos, end_pos, num_lines_tot, num_lines_abs])
                        all_docs.append(one_doc)
                        break
                    elif a_line.find('<ABSTRACT>') >= 0:
                        num_lines_abs = 0
                        abstract_content = ''
                        a_line, line_cnt, char_cnt = line_char_counting_reader(f, line_cnt, char_cnt)
                        # abstracts have multiple lines -- in some cases they have multiple paragraphs
                        # for the time being, and for our inverted index purpose, we collapse all of them into a string
                        while a_line.find('</ABSTRACT>') == -1:
                            num_lines_abs = num_lines_abs + 1
                            abstract_content = abstract_content + ' ' + a_line
                            a_line, line_cnt, char_cnt = line_char_counting_reader(f, line_cnt, char_cnt)
                        one_doc.append(abstract_content.strip())
                    elif a_line.find('<NUMBER>') >= 0:
                        b = a_line.find('<NUMBER>')
                        e = a_line.find('</NUMBER>')
                        one_doc.append(a_line[(b+len('<NUMBER>')):e].strip())
                    elif a_line.find('<ORDER>') >= 0:
                        b = a_line.find('<ORDER>')
                        e = a_line.find('</ORDER>')
                        one_doc.append(a_line[(b+len('<ORDER>')):e].strip())
                    elif a_line.find('<TITLE>') >= 0:
                        txt_pos = char_cnt - len(a_line) + len('<TITLE>')
                        b = a_line.find('<TITLE>')
                        e = a_line.find('</TITLE>')
                        one_doc.append(a_line[(b+len('<TITLE>')):e].strip())
                    elif a_line.find('<AUTHOR>') >= 0:
                        b = a_line.find('<AUTHOR>')
                        e = a_line.find('</AUTHOR>')
                        one_doc.append(a_line[(b+len('<AUTHOR>')):e].strip())
                    elif a_line.find('<YEAR>') >= 0:
                        b = a_line.find('<YEAR>')
                        e = a_line.find('</YEAR>')
                        one_doc.append(a_line[(b+len('<YEAR>')):e].strip())
                    elif a_line.find('<INSTITUTION>') >= 0:
                        b = a_line.find('<INSTITUTION>')
                        e = a_line.find('</INSTITUTION>')
                        one_doc.append(a_line[(b+len('<INSTITUTION>')):e].strip())
                    elif a_line.find('<DESCRIPTORS>') >= 0:
                        b = a_line.find('<DESCRIPTORS>')
                        e = a_line.find('</DESCRIPTORS>')
                        one_doc.append(a_line[(b+len('<DESCRIPTORS>')):e].strip())
                    elif a_line.find('<ADVISER>') >= 0:
                        b = a_line.find('<ADVISER>')
                        e = a_line.find('</ADVISER>')
                        one_doc.append(a_line[(b+len('<ADVISER>')):e].strip())
                    elif a_line.find('<CLASSIFICATIONS>') >= 0:
                        b = a_line.find('<CLASSIFICATIONS>')
                        e = a_line.find('</CLASSIFICATIONS>')
                        one_doc.append(a_line[(b+len('<CLASSIFICATIONS>')):e].strip())
                    else:
                        print('Unknown format!')
                        f.close()
                        return
            elif len(a_line) > 0:
                continue
            else:
                break

        f.close()

    # save the parsing result as a binary file
    with open(corpora_file, 'wb') as f:
        pickle.dump([all_docs, line_cnt, char_cnt], f)


def generate_simple_corpora(corpora_file, noise_words_path, corpora_simple_wo_nw):
    # read binary corpora; all_docs contains the parsed document list
    with open(corpora_file, 'rb') as f:
        all_docs, _, _ = pickle.load(f)

    print('Loading noise (stop) word list...')

    # prepare noise words
    if noise_words_path == '':
        stop_words = set(stopwords.words('english'))
    else:
        with open(noise_words_path, 'rt') as f:
            stop_words = f.read()
        stop_words = word_tokenize(stop_words)

    print('Building stemmed corpora without noise (stop) words...')

    # we are preparing a document collection -- without noise (stop) words + stemmed
    all_docs_simple_wo_nw = []
    for a_doc in all_docs:
        one_doc_wo_nw = [a_doc[0], a_doc[1]]  # file number, doc number

        str_title = word_tokenize(a_doc[3])  # title
        filtered_title = []
        for w in str_title:
            if w not in stop_words:
                filtered_title.append(db_config.g_stemmer.stem(w))

        str_abstract = word_tokenize(a_doc[10])  # abstract
        filtered_abstract = []
        for w in str_abstract:
            if w not in stop_words:
                filtered_abstract.append(db_config.g_stemmer.stem(w))

        one_doc_wo_nw.append(filtered_title)
        one_doc_wo_nw.append(filtered_abstract)

        # BEGIN: ADDED THIS TIME
        list_descriptors = [x.strip() for x in a_doc[7].split(';')]  # descriptor
        one_doc_wo_nw.append(list_descriptors)
        # END  : ADDED THIS TIME

        all_docs_simple_wo_nw.append(one_doc_wo_nw)

    # get unique list of words and sort it out
    print('Obtaining unique word list. This will take some time...')

    # collect tokenized words from title and abstract, of ALL documents, into a single set of list and sort them out
    all_words_wo_nw = []
    for a_doc in all_docs_simple_wo_nw:
        all_words_wo_nw = list(set(all_words_wo_nw + list(set(a_doc[2] + a_doc[3]))))
    all_words_wo_nw.sort()

    # ignore keywords that do not start with an alphabet
    reg_ex = re.compile('^[a-zA-Z].*')
    all_words_wo_nw = [w for w in all_words_wo_nw if reg_ex.search(w)]

    # write!
    with open(corpora_simple_wo_nw, 'wb') as f:
        pickle.dump([all_docs_simple_wo_nw, all_words_wo_nw], f)

    print('Done!')


def compute_inverted_index(simple_binary_copora, inverted_index_path):
    with open(simple_binary_copora, 'rb') as f:
        [all_docs, all_words] = pickle.load(f)

    print('Building inverted index. This will take some time...')

    # now that we have list of words and list of documents, remaining task is to do the counting and build posting
    # my implementation here is pretty naive; there should be a better, more efficient algorithm!
    cnt = 0
    inv_dic = []
    for w in all_words:
        total_doc_cnt = 0
        total_freq_cnt = 0

        doc_list = []
        for a_doc in all_docs:
            one_freq_cnt = a_doc[2].count(w) + a_doc[3].count(w)
            if one_freq_cnt > 0:
                total_doc_cnt = total_doc_cnt + 1
                total_freq_cnt = total_freq_cnt + one_freq_cnt
                doc_list.append([a_doc[1], one_freq_cnt])

        inv_dic.append([w, total_doc_cnt, total_freq_cnt, doc_list])
        # progress output
        cnt = cnt + 1
        if cnt % 100 == 0:
            print('Processed ' + str(cnt) + '/' + str(len(all_words)))

    keywords = [kw_head[0] for kw_head in inv_dic]  # quick index for lookup purpose later

    # write binary
    with open(inverted_index_path, 'wb') as f:
        pickle.dump([inv_dic, keywords], f)

    print('Done!')


def main():
    # Part 1: We do Assignment 1 again. This time, though, we consider descriptor tag as well.
    # 1. create files.d file
    if not os.path.exists(db_config.g_db_file_path):
        create_files_database_from_path(db_config.g_xml_file_path, db_config.g_db_file_path)

    # 2. create documents.d file (binary)
    if not os.path.exists(db_config.g_corpora_file):
        parse_ait_corpora(db_config.g_db_file_path, db_config.g_corpora_file)

    # 3. read stop word and build simplified corpora with and without stop words
    #    simplified corpora only contains: file#, doc#, abstract
    if not (os.path.exists(db_config.g_corpora_simple)):
        generate_simple_corpora(db_config.g_corpora_file, db_config.g_noise_words_path, db_config.g_corpora_simple)

    # 4. using the simplified corpora, compute inverted index (w/o noise words)
    if not os.path.exists(db_config.g_inverted_index):
        compute_inverted_index(db_config.g_corpora_simple, db_config.g_inverted_index)


main()
