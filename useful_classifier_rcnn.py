#! /usr/bin/env python
from __future__ import print_function
import re
import jieba
import numpy as np
import pandas as pd
import pickle
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras import layers
from keras import models
from keras import regularizers
from keras import backend as K

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
    filters = [2, 3]
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
    output = layers.Dense(classification_type, activation='softmax')(x)
    model = models.Model(sentence_input, output)
    return model


def clean_corpus(txts):
    new_txts = []
    for txt in txts:
        txt = re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9^\s]', ' ', str(txt))
        seg_list = jieba.cut(txt)
        line = ' '.join(seg_list)
        new_txts.append(line)
    return new_txts


corpus_info = [{'name': 'movie', 'text': 'comment', 'sent': 'like', 'chinese': True},
               {'name': 'yelp', 'text': 'text', 'sent': 'stars', 'chinese': False}]

model = build_text_rcnn(max_words=MAX_NB_WORDS,
                        maxlen=MAX_SEQUENCE_LENGTH,
                        embedding_dim=EMBEDDING_DIM,
                        classification_type=2)

model.summary()

for corpus in []:
    df_1 = pd.read_csv(open('./data/{}_unuseful.csv'.format(corpus['name']), 'r', encoding='UTF-8'))
    text_1 = df_1[corpus['text']]
    df_2 = pd.read_csv(open('./data/{}_useful.csv'.format(corpus['name']), 'r', encoding='UTF-8'))
    text_2 = df_2[corpus['text']]

    texts = list(text_1) + list(text_2)
    if corpus['chinese']:
        texts = clean_corpus(texts)
    labels = [0 for i in range(text_1.shape[0])] + [1 for i in range(text_2.shape[0])]

    txt_tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    txt_tokenizer.fit_on_texts(texts)
    txt_seq = txt_tokenizer.texts_to_sequences(texts)
    X = pad_sequences(txt_seq, maxlen=MAX_SEQUENCE_LENGTH)
    Y = to_categorical(np.asarray(labels))

    X_, X_predict, Y_, Y_predict = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)

    model = build_text_rcnn(max_words=MAX_NB_WORDS,
                            maxlen=MAX_SEQUENCE_LENGTH,
                            embedding_dim=EMBEDDING_DIM,
                            classification_type=2)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()

    history = model.fit(X_, Y_, validation_data=(X_predict, Y_predict),
                        nb_epoch=5, batch_size=100, verbose=True)
    with open('./history/result_{}.pkl'.format(corpus['name']), 'wb') as f:
        pickle.dump(history.history, f)
    _predict = model.predict(X_predict, batch_size=100)
    with open('./model/test_{}.plk'.format(corpus['name']), 'wb') as f:
        pickle.dump(_predict, f)
    del model
