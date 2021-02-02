import jieba.posseg
import re

NOUN_POS = ['n', 'nr', 'ns', 'nt', 'nz']
VERB_POS = ['v', 'vg']
ADJ_POS = ['a', 'an', 'ad', 'Ag']
ADV_POS = ['d', 'dg']

FIRST_PERSION_WORDS = ['我', '我们', '咱', '咱们']
SECOND_PERSON_WORDS = ['你', '你们', '您', '您们']
THIRD_PERSON_WORDS = ['她', "她们", '他', '他们', '它', '它们']


def cut_sent(para):
    para = re.sub('([。！？.!?\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？.!?\?][”’])([^，。！？.!?\?])', r'\1\n\2', para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")


def process_stylistic_feature(text):

    sentences = cut_sent(text)

    number_of_sentences = len(sentences)
    max_sentence_length = -1
    min_sentence_length = 100000000
    average_sentence_length = 0
    number_of_question_sentences = 0
    number_of_exclamatory_sentences = 0

    for sentence in sentences:
        if len(sentence) > 0:
            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence)
            if len(sentence) < min_sentence_length:
                min_sentence_length = len(sentence)
            if sentence[-1] == '?' or sentence[-1] == '？':
                number_of_question_sentences += 1/len(sentence)
            if sentence[-1] == '!' or sentence[-1] == '！':
                number_of_exclamatory_sentences += 1/len(sentence)
            average_sentence_length += len(sentence)/number_of_sentences

    number_of_nouns = 0
    number_of_verbs = 0
    number_of_adj = 0
    number_of_adv = 0
    number_of_func = 0

    number_of_long_words = 0

    number_of_first_person_pronoun = 0
    number_of_second_person_pronoun = 0
    number_of_third_person_pronoun = 0

    total_words = []

    for sentence in sentences:
        out = jieba.posseg.cut(sentence.strip())
        for x in out:
            total_words.append(x.word)
            if x.flag in NOUN_POS:
                number_of_nouns += 1
            elif x.flag in ADJ_POS:
                number_of_adj += 1
            elif x.flag in ADV_POS:
                number_of_adv += 1
            elif x.flag in VERB_POS:
                number_of_verbs += 1
            else:
                number_of_func += 1

            if x.word in FIRST_PERSION_WORDS:
                number_of_first_person_pronoun += 1
            if x.word in SECOND_PERSON_WORDS:
                number_of_second_person_pronoun += 1
            if x.word in THIRD_PERSON_WORDS:
                number_of_third_person_pronoun += 1

            if len(x.word) >= 3:
                number_of_long_words += 1

    number_of_words = len(total_words)
    number_of_distinct_words = len(set(total_words))

    if len(total_words) == 0:
        average_word_length = 0
    else:
        average_word_length = sum([len(word) for word in total_words])/len(total_words)

    stylistic_feature = {'number_of_words': number_of_words,
                         'number_of_distinct_words': number_of_distinct_words,
                         'average_word_length': average_word_length,
                         'number_of_long_words': number_of_long_words,
                         'number_of_nouns': number_of_nouns,
                         'number_of_verbs': number_of_verbs,
                         'number_of_adj': number_of_adj,
                         'number_of_adv': number_of_adv,
                         'number_of_func': number_of_func,
                         'number_of_first_person_pronoun': number_of_first_person_pronoun,
                         'number_of_second_person_pronoun': number_of_second_person_pronoun,
                         'number_of_third_person_pronoun': number_of_third_person_pronoun,
                         'number_of_sentences': number_of_sentences,
                         'max_sentence_length': max_sentence_length,
                         'min_sentence_length': min_sentence_length,
                         'average_sentence_length': average_sentence_length,
                         'number_of_question_sentences': number_of_question_sentences,
                         'number_of_exclamatory_sentences': number_of_question_sentences}
    return stylistic_feature


if __name__ == '__main__':
    text = '我爱咱们的祖国。我和我的祖国一刻都不能分割. 最浪漫的事就是一起和你变老'
    st_feauture = process_stylistic_feature(text)
    for key in st_feauture:
        print(key, st_feauture[key])
    print('totol stylistic feautures: ' + str(len(st_feauture.keys())))
