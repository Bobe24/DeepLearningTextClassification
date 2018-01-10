# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com
# Function: preprocess raw data and pretrain word embedding

from gensim.models import Word2Vec, Doc2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
import os
import logging

from config import train_path, test_path, label_dict_path, unlabeled_path


def get_unlabeled_data(unlabeled_path):
    unlabeled_data = list()
    for root, dirs, files in os.walk(unlabeled_path):
        for file in files:
            if file[-4:] == '.txt':
                file_path = unlabeled_path + file
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not len(line) or not line.startswith("<content>"):
                            continue
                        line = line.lstrip("<content>")
                        line = line.rstrip("</content>")
                        if len(line):
                            unlabeled_data.append(line)
    return unlabeled_data


def embedding_train(sentences):
    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Set values for various parameters
    num_features = 100  # Word vector dimensionality
    min_word_count = 5  # Minimum word count
    num_workers = 3  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    # Initialize and train the model (this will take some time)
    model = Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1)
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "../data/100features_5minwords_10context"
    model.save(model_name)


if __name__ == "__main__":
    unlabeled_data = get_unlabeled_data(unlabeled_path)
    # set the punctuation
    tokenizer = "[.。!！?？;；]"
    sentences = []  # Initialize an empty list of sentences
    # Parsing sentences from unlabeled set
    for news in unlabeled_data:
        sentences += KaggleWord2VecUtility.review_to_sentences(news, tokenizer)
    # train word embedding
    embedding_train(sentences)
