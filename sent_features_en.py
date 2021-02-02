import pandas as pd
import nltk
from nltk import word_tokenize
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
import string
import os

stop = stopwords.words("english") + list(string.punctuation)

n = ['NN', 'NNP', 'NNPS', 'NNS', 'UH']
v = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
a = ['JJ', 'JJR', 'JJS']
r = ['RB', 'RBR', 'RBS', 'RP', 'WRB']


def get_sent_pos_token(token):
    if token in n:
        return 'n'
    if token in v:
        return 'v'
    if token in a:
        return 'a'
    if token in r:
        return 'r'
    return ''


def process_sent_features(text):
    sentences = tokenize.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    number_of_sentences = len(sentences)
    number_of_words = sum([len(sentence) for sentence in sentences])

    number_of_positive_words = 0
    number_of_negative_words = 0
    number_of_objective_sentences = 0
    number_of_subjective_sentences = 0

    for sentence in sentences:
        pos_tags = nltk.pos_tag(sentence)
        sent_scores = []
        for word, pos in pos_tags:
            sent_pos_token = get_sent_pos_token(pos)
            m = list(swn.senti_synsets(word, sent_pos_token))
            s = 0
            ra = 0
            score = 0
            if len(m) > 0:
                for j in range(len(m)):
                    s += (m[j].pos_score() - m[j].neg_score()) / (j + 1)
                    ra += 1 / (j + 1)
                score = s / ra
            sent_scores.append(score)
            if score > 0:
                number_of_positive_words += 1
            if score < 0:
                number_of_negative_words += 1
        if len(set(sent_scores)) > 1:
            number_of_subjective_sentences += 1
        else:
            number_of_objective_sentences += 1

    sentiment_feature = {'number_of_positive_words': number_of_positive_words,
                         'number_of_negative_words': number_of_negative_words,
                         'number_of_objective_sentences': number_of_objective_sentences,
                         'number_of_subjective_sentences': number_of_subjective_sentences,
                         'positive_ratio': number_of_positive_words / number_of_words,
                         'negative_ratio': number_of_negative_words / number_of_words,
                         'subjective_ratio': number_of_subjective_sentences / number_of_sentences,
                         'objective_ratio': number_of_objective_sentences / number_of_sentences}
    return sentiment_feature


if __name__ == '__main__':
    pos_index = {}
    f = open(os.path.join('./', 'demo_review_text.dat'), encoding='UTF-8')
    review_text = f.read()
    print(review_text)
    f.close()
    st_feauture = process_sent_features(review_text)
    for key in st_feauture:
        print(key, st_feauture[key])
    print('totol stylistic feautures: ' + str(len(st_feauture.keys())))
    print(list(st_feauture.values()))


