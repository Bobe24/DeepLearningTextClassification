# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com
# Function: main function, call different method

import sys
import time
from news_lstm import process_data, get_pre_embedding_matrix, train_lstm_word, train_lstm_char, train_lstm_w_c,\
    train_lstm_word_extend, process_data_extend

MAX_NB_WORDS = 60000  # 最常见的词的数量
MAX_NB_CHARS = 4000  # 最常见的字的数量
embedding_dim = 100  # 词向量/字向量维度


def main(method):
    if method == 'lstm_word':
        word_X_train, word_Y_train, word_X_test, word_Y_test, word_index = process_data(is_char=False)
        word_embedding, word_num_words = get_pre_embedding_matrix(word_index, embedding_dim=embedding_dim,
                                                                  MAX_NB=MAX_NB_WORDS)
        train_lstm_word(word_index, word_embedding, word_X_train, word_Y_train, word_X_test, word_Y_test)
    elif method == 'lstm_char':
        char_X_train, char_Y_train, char_X_test, char_Y_test, char_index = process_data(is_char=True)
        char_embedding, char_num_words = get_pre_embedding_matrix(char_index, embedding_dim=embedding_dim,
                                                                  MAX_NB=MAX_NB_CHARS)
        train_lstm_char(char_index, char_embedding, char_X_train, char_Y_train, char_X_test, char_Y_test)
    elif method == 'lstm_w_c':
        word_X_train, word_Y_train, word_X_test, word_Y_test, word_index = process_data(is_char=False)
        char_X_train, char_Y_train, char_X_test, char_Y_test, char_index = process_data(is_char=True)
        word_embedding, word_num_words = get_pre_embedding_matrix(word_index, embedding_dim=embedding_dim,
                                                                  MAX_NB=MAX_NB_WORDS)
        char_embedding, char_num_words = get_pre_embedding_matrix(char_index, embedding_dim=embedding_dim,
                                                                  MAX_NB=MAX_NB_CHARS)
        train_lstm_w_c(word_index, word_embedding, word_X_train, word_Y_train, word_X_test, word_Y_test,
                       char_index, char_embedding, char_X_train, char_Y_train, char_X_test, char_Y_test)
    elif method == "lstm_w_e":
        X_train, X_train_extend, Y_train, X_test, X_test_extend, Y_test, word_index, word_index_extend = \
            process_data_extend(is_char=False)
        word_embedding, word_num_words = get_pre_embedding_matrix(word_index, embedding_dim=embedding_dim,
                                                                  MAX_NB=MAX_NB_WORDS)
        word_embedding_extend, word_num_words_extend = get_pre_embedding_matrix(word_index_extend,
                                                                                embedding_dim=embedding_dim,
                                                                                MAX_NB=MAX_NB_CHARS)
        train_lstm_word_extend(word_index, word_embedding, X_train, Y_train, X_test, Y_test,
                               word_index_extend, word_embedding_extend, X_train_extend, X_test_extend)
    else:
        print("No such method!")
        return None


if __name__ == "__main__":
    method = sys.argv[1]
    start = time.time()
    main(method=method)
    end = time.time()
    elapsed = end - start
    print("Time taken: ", elapsed, "seconds.")
