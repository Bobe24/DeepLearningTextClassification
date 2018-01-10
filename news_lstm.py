# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com
# Function: process data, train and test different method

import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Bidirectional, Merge, Conv1D, MaxPooling1D, Embedding, Dense, Input, Flatten, Dropout
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelBinarizer
from gensim.models.doc2vec import LabeledSentence, TaggedDocument, Doc2Vec
from keras.preprocessing.text import Tokenizer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

from KaggleWord2VecUtility import KaggleWord2VecUtility
from config import train_path, test_path, label_dict_path, word_model_path, char_model_path, word_pkl_path, \
    char_pkl_path, pv_dm_model_path, pv_dbow_model_path, word_lstm_model_path, cnn_model_path, c_lstm_model_path, \
    lstm_char_model_path, lstm_sentence_model_path, cnn_char_model_path, cnn_sentence_model_path, lstm_w_c_model_path, \
    lstm_w_s_model_path, lstm_c_s_model_path, cnn_w_c_model_path, cnn_w_s_model_path, cnn_c_s_model_path, \
    c_cnn_model_path


# 参数设置
embedding_dim = 100  # 词向量/字向量维度
maxlen = 70  # 文本保留的最大长度
batch_size = 128
n_epoch = 50
input_length = maxlen
dropout = 0.4
size = 500  # 句子向量纬度
output_dim = 64
train_size = 47866
test_size = 15955
MAX_NB_WORDS = 60000  # 最常见的词的数量
MAX_NB_CHARS = 4000  # 最常见的字的数量


# 将训练集和测试集表示成index并处理成相同长度
def process_data(is_char=False, is_extend=False):
    # 加载训练集和测试集
    x_train, y_train = get_data(train_path, convert_label=False)
    x_test, y_test = get_data(test_path, convert_label=False)
    # 将文本转化为句子，每个句子是由词组成的list
    train_sentences = []  # Initialize an empty list of sentences
    test_sentences = []  # Initialize an empty list of sentences
    for news in x_train:
        if len(news) > 0:
            train_sentences.append(
                " ".join(KaggleWord2VecUtility.review_to_wordlist(news, is_char=is_char,
                                                       remove_stopwords=True)))
    for news in x_test:
        if len(news) > 0:
            test_sentences.append(
                " ".join(KaggleWord2VecUtility.review_to_wordlist(news, is_char=is_char,
                                                                  remove_stopwords=True)))
    # 对短文本进行扩充
    if is_extend:
        train_sentences = get_similar_words(train_sentences)
        test_sentences = get_similar_words(test_sentences)

    all_text = train_sentences
    all_text.extend(test_sentences)
    # 将句子表示成数字索引
    sequences, word_index = gen_word_index(all_text, is_char=is_char)
    # Padding sequences
    if is_extend:
        data = sequence.pad_sequences(sequences, maxlen=70)
    else:
        data = sequence.pad_sequences(sequences, maxlen=maxlen)
    X_train = data[:train_size]
    X_test = data[train_size:]
    # label转成numpy数组
    Y_train = np.array(y_train)
    Y_test = np.array(y_test)
    # one-hot
    encoder = LabelBinarizer().fit(Y_train)
    Y_train = encoder.transform(Y_train)
    Y_test = encoder.transform(Y_test)
    return X_train, Y_train, X_test, Y_test, word_index


def process_data_extend(is_char=False):
    # 加载训练集和测试集
    x_train, y_train = get_data(train_path, convert_label=False)
    x_test, y_test = get_data(test_path, convert_label=False)
    # 将文本转化为句子，每个句子是由词组成的list
    train_sentences = []  # Initialize an empty list of sentences
    test_sentences = []  # Initialize an empty list of sentences
    for news in x_train:
        if len(news) > 0:
            train_sentences.append(
                " ".join(KaggleWord2VecUtility.review_to_wordlist(news, is_char=is_char,
                                                                  remove_stopwords=True)))
    for news in x_test:
        if len(news) > 0:
            test_sentences.append(
                " ".join(KaggleWord2VecUtility.review_to_wordlist(news, is_char=is_char,
                                                                  remove_stopwords=True)))

    # 对短文本进行扩充
    train_extend = get_similar_words(train_sentences)
    test_extend = get_similar_words(test_sentences)

    # 转为数字索引形式
    all_text = train_sentences
    all_text.extend(test_sentences)
    sequences, word_index = gen_word_index(all_text, is_char=is_char)
    data = sequence.pad_sequences(sequences, maxlen=maxlen)
    X_train = data[:train_size]
    X_test = data[train_size:]
    all_text_extend = train_extend
    all_text_extend.extend(test_extend)
    sequences, word_index_extend = gen_word_index(all_text_extend, is_char=is_char)
    data_extend = sequence.pad_sequences(sequences, maxlen=maxlen)
    X_train_extend = data_extend[:train_size]
    X_test_extend = data_extend[train_size:]

    # label转成numpy数组
    Y_train = np.array(y_train)
    Y_test = np.array(y_test)
    # one-hot
    encoder = LabelBinarizer().fit(Y_train)
    Y_train = encoder.transform(Y_train)
    Y_test = encoder.transform(Y_test)
    return X_train, X_train_extend, Y_train, X_test, X_test_extend, Y_test, word_index, word_index_extend


# 加载训练集 测试集 并将label转换为数字
def get_data(path, convert_label=False):
    data = pd.read_csv(path, sep='\t', header=None, names=['label', 'news'])
    x = data['news']
    y = data['label']
    f = open(label_dict_path, 'r')
    label_dict = f.readlines()
    f.close()
    label_dict = eval(label_dict[0])
    # print((label_dict['it']))
    y_list = list(y)
    y_num = list()
    for i in range(len(y_list)):
        y_num.append(label_dict[y_list[i]])
    if convert_label:
        key = list(set(y))
        value = range(len(key))
        label_dict = dict(zip(key, value))
        f = open('../data/sample_data/label_dict.txt', 'w')
        label_dict_repr = repr(label_dict)
        f.writelines(label_dict_repr)
        f.close()
    return list(x), y_num


# 得到词向量 和词的数量
def get_pre_embedding_matrix(word_index, embedding_dim, MAX_NB):
    num_words = min(MAX_NB, len(word_index))
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    w2v_model = Word2Vec.load(word_model_path)
    index2word_set = set(w2v_model.wv.index2word)
    for word, i in word_index.items():
        # if i >= MAX_NB:
        #     continue
        if word in index2word_set:
            embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')
    return embedding_matrix, num_words


# 得到最相似的词
def get_similar_words(sentences):
    w2v_model = Word2Vec.load(word_model_path)
    index2word_set = set(w2v_model.wv.index2word)
    sentences_extend = []
    for news in sentences:
        news_list = news.split()
        news_extend = []
        for word in news_list:
            if word in index2word_set:
                cosine_sim = w2v_model.most_similar(word, topn=1)[0][1]
                if cosine_sim > 0.7:
                    similar_word = w2v_model.most_similar(word, topn=1)[0][0]
                    news_extend.append(similar_word)
        if len(news_extend) == 0:
            news_extend = news_list
        sentences_extend.append(" ".join(news_extend))
    return sentences_extend


# 将句子表示成数字索引
def gen_word_index(all_texts, is_char):
    # if is_char:
    #     tokenizer = Tokenizer(num_words=MAX_NB_CHARS)
    # else:
    #     tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return sequences, word_index


def train_lstm_word(word_index, word_embedding, word_X_train, word_Y_train, word_X_test, word_Y_test, method):
    if method == 'lstm_w_e':
        input_len = 70
    else:
        input_len = 70
    embedding_layer_word = Embedding(
        len(word_index) + 1,
        # word_num_words,
        embedding_dim,
        weights=[word_embedding],
        input_length=input_len,
        trainable=True
        # dropout=0.4,
        # mask_zero=True
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # 0.7896
    model_trained_path = word_lstm_model_path
    # 创建模型...word'
    model = Sequential()
    model.add(embedding_layer_word)
    model.add(LSTM(units=128, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(32, activation='softmax'))
    model.summary()
    # 编译模型
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练
    model.fit(word_X_train, word_Y_train, batch_size=batch_size, epochs=50,
              validation_data=(word_X_test, word_Y_test), callbacks=[early_stopping])
    # 评估
    score, acc = model.evaluate(word_X_test, word_Y_test, batch_size=batch_size)
    predict_Y_test = model.predict_classes(word_X_test, batch_size=batch_size)
    predict_Y_path = "../data/predict_class/" + method + ".txt"
    f = open(predict_Y_path, 'w')
    for item in predict_Y_test:
        f.write(str(item) + '\n')
    f.close()
    print('Test score:', score)
    print('Test accuracy:', acc)
    model.save(model_trained_path)


def train_lstm_char(char_index, char_embedding, char_X_train, char_Y_train, char_X_test, char_Y_test):
    embedding_layer_char = Embedding(
        len(char_index) + 1,
        # char_num_words,
        embedding_dim,
        weights=[char_embedding],
        input_length=input_length,
        trainable=True
        # dropout=0.4,
        # mask_zero=True
    )
    model_trained_path = lstm_char_model_path

    model = Sequential()
    model.add(embedding_layer_char)
    model.add(LSTM(units=128, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(32, activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'f1score', 'precision', 'recall'])
    model.fit(char_X_train, char_Y_train, batch_size=batch_size, epochs=n_epoch,
              validation_data=(char_X_test, char_Y_test))
    score, acc = model.evaluate(char_X_test, char_Y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    model.save(model_trained_path)


def train_lstm_w_c(word_index, word_embedding, word_X_train, word_Y_train, word_X_test, word_Y_test,
                   char_index, char_embedding, char_X_train, char_Y_train, char_X_test, char_Y_test):
    embedding_layer_word = Embedding(
        len(word_index) + 1,
        # word_num_words,
        embedding_dim,
        weights=[word_embedding],
        input_length=input_length,
        trainable=True
        # dropout=0.4,
        # mask_zero=True
    )
    embedding_layer_char = Embedding(
        len(char_index) + 1,
        # char_num_words,
        embedding_dim,
        weights=[char_embedding],
        input_length=input_length,
        trainable=True
        # dropout=0.4,
        # mask_zero=True
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model_trained_path = lstm_w_c_model_path
    drop_out = 0.5
    # 词向量
    word_lstm = Sequential()
    word_lstm.add(embedding_layer_word)
    word_lstm.add(LSTM(units=128, dropout=drop_out, recurrent_dropout=drop_out))
    word_lstm.add(Dense(128))
    word_lstm.add(Dropout(drop_out))
    # 字向量
    char_lstm = Sequential()
    char_lstm.add(embedding_layer_char)
    char_lstm.add(LSTM(units=128, dropout=drop_out, recurrent_dropout=drop_out))
    char_lstm.add(Dense(128))
    char_lstm.add(Dropout(drop_out))
    # merge两个向量
    model = Sequential()
    model.add(Merge([word_lstm, char_lstm], mode='concat', concat_axis=-1))
    model.add(Dense(32, activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([word_X_train, char_X_train], word_Y_train, batch_size=32, epochs=n_epoch,
              validation_data=([word_X_test, char_X_test], word_Y_test), callbacks=[early_stopping])
    score, acc = model.evaluate([word_X_test, char_X_test], word_Y_test, batch_size=32)
    predict_Y_test = model.predict_classes([word_X_test, char_X_test], batch_size=32)
    predict_Y_path = "../data/predict_class/lstm_w_c.txt"
    f = open(predict_Y_path, 'w')
    for item in predict_Y_test:
        f.write(str(item) + '\n')
    f.close()
    print('Test score:', score)
    print('Test accuracy:', acc)
    model.save(model_trained_path)


def train_lstm_word_extend(word_index, word_embedding, word_X_train, word_Y_train, word_X_test, word_Y_test,
                              word_index_extend, word_embedding_extend, X_train_extend, X_test_extend):
    embedding_layer_word = Embedding(
        len(word_index) + 1,
        # word_num_words,
        embedding_dim,
        weights=[word_embedding],
        input_length=input_length,
        trainable=True
        # dropout=0.4,
        # mask_zero=True
    )
    embedding_layer_char = Embedding(
        len(word_index_extend) + 1,
        # char_num_words,
        embedding_dim,
        weights=[word_embedding_extend],
        input_length=input_length,
        trainable=True
        # dropout=0.4,
        # mask_zero=True
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model_trained_path = lstm_w_c_model_path
    # 词向量
    word_lstm = Sequential()
    word_lstm.add(embedding_layer_word)
    word_lstm.add(LSTM(units=128, dropout=0.4, recurrent_dropout=0.4))
    word_lstm.add(Dense(128))
    word_lstm.add(Dropout(0.4))
    # 字向量
    char_lstm = Sequential()
    char_lstm.add(embedding_layer_char)
    char_lstm.add(LSTM(units=128, dropout=0.4, recurrent_dropout=0.4))
    char_lstm.add(Dense(128))
    char_lstm.add(Dropout(0.4))
    # merge两个向量
    model = Sequential()
    model.add(Merge([word_lstm, char_lstm], mode='concat', concat_axis=-1))
    model.add(Dense(32, activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([word_X_train, X_train_extend], word_Y_train, batch_size=batch_size, epochs=n_epoch,
              validation_data=([word_X_test, X_test_extend], word_Y_test), callbacks=[early_stopping])
    score, acc = model.evaluate([word_X_test, X_test_extend], word_Y_test, batch_size=batch_size)
    predict_Y_test = model.predict_classes([word_X_test, X_test_extend], batch_size=batch_size)
    predict_Y_path = "../data/predict_class/lstm_w_e.txt"
    f = open(predict_Y_path, 'w')
    for item in predict_Y_test:
        f.write(str(item) + '\n')
    f.close()
    print('Test score:', score)
    print('Test accuracy:', acc)
    model.save(model_trained_path)
