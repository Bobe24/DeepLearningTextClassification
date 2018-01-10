# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Modified by Bobe_24@126.com
# Function: KaggleWord2VecUtility is a utility class for processing raw news text into segments for further learning

import re
from string import punctuation as punctuation_en
import jieba


class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw news text into segments for further learning"""

    @staticmethod
    def review_to_wordlist(review, is_char=False, remove_stopwords=False):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        # 1. Remove punctuation and space
        review_text = review
        # 自定义过滤字符
        pattern = u'[\s+’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        review_text = re.sub(pattern, "", review_text)
        review_text = re.sub(r'[{}]+'.format(punctuation_en), "", review_text)

        # 2. Optionally remove stop words (false by default)
        if remove_stopwords:
            # review_text = re.sub("[a-zA-Z]", "", review_text)
            review_text = re.sub("[0-9]", "", review_text)
            if is_char:
                words = list(review_text)
            else:
                words = jieba.lcut(review_text)
            path = r"../data/stopwords.txt"
            with open(path) as file:
                stopwords = file.readlines()
            # stopwords = [stopword.decode('gbk', 'ignore').strip('\n') for stopword in stopwords]
            stopwords = [stopword.strip('\n') for stopword in stopwords]
            stops = set(stopwords)
            words = [w for w in words if not w in stops]
        else:
            if is_char:
                words = list(review_text)
            else:
                words = jieba.lcut(review_text)

        # 3. Return a list of words
        return words

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences(review, tokenizer, remove_stopwords=False):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        raw_sentences = re.split(tokenizer, review.strip())
        is_char = False
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append(KaggleWord2VecUtility.review_to_wordlist(raw_sentence, is_char=is_char,
                                                                          remove_stopwords=remove_stopwords))
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
