import re
import jieba

with open('./sent_dic/正面情感词语（中文）.txt', 'r', encoding='GBK') as f:
    lines_0 = f.readlines()

with open('./sent_dic/正面评价词语（中文）.txt', 'r', encoding='GBK') as f:
    lines_1 = f.readlines()

with open('./sent_dic/负面情感词语（中文）.txt', 'r', encoding='GBK') as f:
    lines_2 = f.readlines()

with open('./sent_dic/负面评价词语（中文）.txt', 'r', encoding='GBK') as f:
    lines_3 = f.readlines()

positive_tokens = [line[:-1].strip() for line in lines_0] + [line[:-1].strip() for line in lines_1]
negative_tokens = [line[:-1].strip() for line in lines_2] + [line[:-1].strip() for line in lines_3]


def cut_sent(para):
    para = re.sub('([。！？.!?\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？.!?\?][”’])([^，。！？.!?\?])', r'\1\n\2', para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")


def process_sent_features(text):
    number_of_words = 0
    number_of_positive_words = 0
    number_of_negative_words = 0
    number_of_objective_sentences = 0
    number_of_subjective_sentences = 0
    sentences = cut_sent(text)
    number_of_sentences = len(sentences)
    for sentence in sentences:
        seg_list = jieba.cut(sentence)
        cur_positive = 0
        cur_negative = 0
        for word in seg_list:
            number_of_words += 1
            if word in positive_tokens:
                cur_positive += 1
            if word in positive_tokens:
                cur_negative += 1

        number_of_positive_words += cur_positive
        number_of_negative_words += cur_negative

        if cur_positive > 0 or cur_negative > 0:
            number_of_subjective_sentences += 1
        else:
            number_of_objective_sentences += 1

    if number_of_words == 0:
        number_of_words = 1

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
    review_text = '我爱我的祖国。我和我的祖国一刻都不能分隔。小明走了。'
    st_feauture = process_sent_features(review_text)
    for key in st_feauture:
        print(key, st_feauture[key])
    print('totol stylistic feautures: ' + str(len(st_feauture.keys())))