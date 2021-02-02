# coding=utf-8
import string
import nltk
from nltk import tokenize
import pandas as pd
import os

NOUN_POS = ['NN', 'NNS', 'NNP', 'NNPS']
VERB_POS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
ADJ_POS = ['JJ', 'JJR', 'JJS']
ADV_POS = ['RB', 'RBR', 'RBS']
FUNCTTION_POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'SYM', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB']

FIRST_PERSION_WORDS = ['i', 'me', 'we', 'us', 'my', 'our', 'mine', 'ours']
SECOND_PERSON_WORDS = ['you', 'your', 'yours']
THIRD_PERSON_WORDS = ['he', "him", 'she', 'her', 'it', 'its', 'they', 'them', 'his', 'hers', 'theirs']


def process_word_feature(s):
    tokens = nltk.word_tokenize(s)
    number_of_words = len(tokens)
    number_of_distinct_words = len(set(tokens))
    average_word_length = sum([len(s) / number_of_words for s in tokens])
    pos_tags = nltk.pos_tag(tokens)
    number_of_long_words = 0

    number_of_noun = 0
    number_of_verb = 0
    number_of_adj = 0
    number_of_adv = 0
    number_of_func = 0

    number_of_first_person_pronoun = 0
    number_of_second_person_pronoun = 0
    number_of_third_person_pronoun = 0

    for word, pos in pos_tags:
        if pos in NOUN_POS:
            number_of_noun += 1
        if pos in VERB_POS:
            number_of_verb += 1
        if pos in ADJ_POS:
            number_of_adj += 1
        if pos in ADV_POS:
            number_of_adv += 1
        if pos in FUNCTTION_POS:
            number_of_func += 1

        if len(word) > 6:
            number_of_long_words += 1

        if word.lower() in FIRST_PERSION_WORDS:
            number_of_first_person_pronoun += 1
        if word.lower() in SECOND_PERSON_WORDS:
            number_of_second_person_pronoun += 1
        if word.lower() in THIRD_PERSON_WORDS:
            number_of_third_person_pronoun += 1

    return number_of_words, number_of_distinct_words, average_word_length, number_of_long_words, \
           number_of_noun, number_of_verb, number_of_adj, number_of_adv, number_of_func,\
           number_of_first_person_pronoun, number_of_second_person_pronoun, number_of_third_person_pronoun


def process_sentence_feature(s):
    sentences = tokenize.sent_tokenize(s)
    number_of_sentences = len(sentences)
    max_sentence_length = -1
    min_sentence_length = 100000000
    average_sentence_length = 0
    number_of_question_sentences = 0
    number_of_exclamatory_sentences = 0

    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            max_sentence_length = len(sentence)
        if len(sentence) < min_sentence_length:
            min_sentence_length = len(sentence)
        if sentence[-1] == '?':
            number_of_question_sentences += 1
        if sentence[-1] == '!':
            number_of_exclamatory_sentences += 1
        average_sentence_length += len(sentence)/number_of_sentences
    return number_of_sentences, max_sentence_length, min_sentence_length, average_sentence_length, number_of_question_sentences, number_of_exclamatory_sentences


def process_stylistic_feature(review_text):
    stylistic_feature = {}
    number_of_words, number_of_distinct_words, average_word_length, number_of_long_words, \
    number_of_noun, number_of_verb, number_of_adj, number_of_adv, number_of_func, \
    number_of_first_person_pronoun, number_of_second_person_pronoun, number_of_third_person_pronoun\
        = process_word_feature(review_text)
    stylistic_feature['number_of_words'] = number_of_words
    stylistic_feature['number_of_distinct_words'] = number_of_distinct_words
    stylistic_feature['average_word_length'] = average_word_length
    stylistic_feature['number_of_long_words'] = number_of_long_words

    stylistic_feature['number_of_noun'] = number_of_noun
    stylistic_feature['number_of_verb'] = number_of_verb
    stylistic_feature['number_of_adj'] = number_of_adj
    stylistic_feature['number_of_adv'] = number_of_adv
    stylistic_feature['number_of_func'] = number_of_func

    stylistic_feature['number_of_first_person_pronoun'] = number_of_first_person_pronoun
    stylistic_feature['number_of_second_person_pronoun'] = number_of_second_person_pronoun
    stylistic_feature['number_of_third_person_pronoun'] = number_of_third_person_pronoun

    number_of_sentences, max_sentence_length, min_sentence_length, average_sentence_length,\
    number_of_question_sentences, number_of_exclamatory_sentences = process_sentence_feature(review_text)
    stylistic_feature['number_of_sentences'] = number_of_sentences
    stylistic_feature['max_sentence_length'] = max_sentence_length
    stylistic_feature['min_sentence_length'] = min_sentence_length
    stylistic_feature['average_sentence_length'] = average_sentence_length
    stylistic_feature['number_of_question_sentences'] = number_of_question_sentences
    stylistic_feature['number_of_exclamatory_sentences'] = number_of_question_sentences
    return stylistic_feature


if __name__ == '__main__':
    pos_index = {}
    f = open(os.path.join('./', 'demo_review_text.dat'), encoding='UTF-8')
    review_text = f.read()
    f.close()
    st_feauture = process_stylistic_feature(review_text)
    for key in st_feauture:
        print(key, st_feauture[key])
    print('totol stylistic feautures: ' + str(len(st_feauture.keys())))
