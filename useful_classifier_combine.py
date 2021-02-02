#! /usr/bin/env python
from __future__ import print_function
import re
import jieba
import numpy as np
import pandas as pd
import pickle
import os
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras import layers
from keras import models
from keras import regularizers
from keras import backend as K

sys.path.append('./')
import score_feature
import sent_features_cn
import sent_features_en
import stylistic_feature_cn
import stylistic_feature_en

SEED = 9
MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100


os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


def build_text_rcnn(max_words, maxlen, embedding_dim, classification_type):
    sentence_input = layers.Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = layers.Embedding(max_words, embedding_dim)(sentence_input)
    x_backwords = layers.GRU(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.32 * 0.1), recurrent_regularizer=regularizers.l2(0.32), go_backwards=True)(embedded_sequences)
    x_backwords_reverse = layers.Lambda(lambda x: K.reverse(x, axes=1))(x_backwords)
    x_fordwords = layers.GRU(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.32 * 0.1), recurrent_regularizer=regularizers.l2(0.32), go_backwards=False)(embedded_sequences)
    x_feb = layers.Concatenate(axis=2)([x_fordwords, embedded_sequences, x_backwords_reverse])
    x_feb = layers.Dropout(0.32)(x_feb)
    # Concatenate后的embedding_size
    dim_2 = K.int_shape(x_feb)[2]
    x_feb_reshape = layers.Reshape((maxlen, dim_2, 1))(x_feb)
    filters = [2, 3, 4, 5]
    conv_pools = []
    for filter in filters:
        conv = layers.Conv2D(filters=300,
                             kernel_size=(filter, dim_2),
                             padding='valid',
                             kernel_initializer='normal',
                             activation='relu',
                             )(x_feb_reshape)
        pooled = layers.MaxPooling2D(pool_size=(maxlen - filter + 1, 1),
                                     strides=(1, 1),
                                     padding='valid',
                                     )(conv)
        conv_pools.append(pooled)

    x = layers.Concatenate()(conv_pools)
    x = layers.Flatten()(x)
    model = models.Model(sentence_input, x)
    return model


def build_non_semantic_DNN(input_shape):
    mlp = models.Sequential()
    mlp.add(layers.Dense(512, input_shape=(input_shape, )))
    mlp.add(layers.BatchNormalization())
    mlp.add(layers.Dense(64, activation='relu'))
    return mlp


def clean_corpus(txts):
    new_txts = []
    for txt in txts:
        txt = re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9^\s]', ' ', str(txt))
        seg_list = jieba.cut(txt)
        line = ' '.join(seg_list)
        new_txts.append(line)
    return new_txts



corpus_info = [
    {'name': 'dianping', 'text': 'Content_review', 'score': 'Rating', 'chinese': True},
               {'name': 'movie', 'text': 'comment', 'score': 'like', 'chinese': True},
               {'name': 'yelp', 'text': 'text', 'score': 'stars', 'chinese': False}]

drops = [None, 0, 1, 2]
for corpus in corpus_info:
    df_1 = pd.read_csv(open('./data/{}_unuseful.csv'.format(corpus['name']), 'r', encoding='UTF-8'))
    text_1 = df_1[corpus['text']]
    score_1 = df_1[corpus['score']]

    df_2 = pd.read_csv(open('./data/{}_useful.csv'.format(corpus['name']), 'r', encoding='UTF-8'))
    text_2 = df_2[corpus['text']]
    score_2 = df_2[corpus['score']]

    texts = list(text_1) + list(text_2)
    scores = list(score_1) + list(score_2)
    # 用户打分特征
    score_features = score_feature.calculate_score_feature(scores)

    # 文体特征与情感特征
    sent_features = []
    stylistic_features = []

    if corpus['chinese']:
        for text in texts:
            text = str(text)
            sent = sent_features_cn.process_sent_features(text)
            sent_features.append(list(sent.values()))
            stylistic = stylistic_feature_cn.process_stylistic_feature(text)
            stylistic_features.append(list(stylistic.values()))
    else:
        for text in texts:
            text = str(text)
            sent = sent_features_en.process_sent_features(text)
            sent_features.append(list(sent.values()))
            stylistic = stylistic_feature_en.process_stylistic_feature(text)
            stylistic_features.append(list(stylistic.values()))

    if corpus['chinese']:
        texts = clean_corpus(texts)

    total_non_semantic_features = [score_features, np.asarray(stylistic_features), np.asarray(sent_features)]
    for drop_index in drops:
        features = None
        if drop_index is not None:
            features = total_non_semantic_features[0:drop_index] + total_non_semantic_features[drop_index + 1:]
        else:
            features = total_non_semantic_features[:]
        features = np.concatenate(tuple(features), axis=1)
        print(features.shape)

        labels = [0 for i in range(text_1.shape[0])] + [1 for i in range(text_2.shape[0])]

        txt_tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
        txt_tokenizer.fit_on_texts(texts)
        txt_seq = txt_tokenizer.texts_to_sequences(texts)
        X = pad_sequences(txt_seq, maxlen=MAX_SEQUENCE_LENGTH)
        Y = to_categorical(np.asarray(labels))

        X_, X_predict, F_, F_predict, Y_, Y_predict = train_test_split(X, features, Y, test_size=0.2, random_state=SEED, stratify=Y)

        model_1 = build_text_rcnn(max_words=MAX_NB_WORDS,
                                maxlen=MAX_SEQUENCE_LENGTH,
                                embedding_dim=EMBEDDING_DIM,
                                classification_type=2)
        model_2 = build_non_semantic_DNN(input_shape=F_.shape[1])

        combinedInput = layers.concatenate([model_1.output, model_2.output])
        x = layers.Dense(2, activation='softmax')(combinedInput)
        model = models.Model(inputs=[model_1.input, model_2.input], outputs=x)

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
        model.summary()

        history = model.fit([X_, F_], Y_, validation_data=([X_predict, F_predict], Y_predict),
                            nb_epoch=4, batch_size=100, verbose=True)
        with open('./history/result_{}_{}.pkl'.format(corpus['name'], str(drop_index)), 'wb') as f:
            pickle.dump(history.history, f)
        _predict = model.predict([X_predict, F_predict], batch_size=100)
        with open('./model/test_{}_{}.plk'.format(corpus['name'],  str(drop_index)), 'wb') as f:
            pickle.dump(_predict, f)
        del model