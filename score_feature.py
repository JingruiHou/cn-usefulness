import numpy as np


def count_x(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count


def calculate_score_feature(scores):
    for idx in range(len(scores)):
        if type(scores[idx]) is not int:
            scores[idx] = 0
    avg = sum(scores)/len(scores)
    shifts = [score-avg for score in scores]
    same = [count_x(scores, score) for score in scores]
    column_1 = np.array(scores).reshape((len(scores), 1))
    column_2 = np.array(shifts).reshape((len(scores), 1))
    column_3 = np.array(same).reshape((len(scores), 1))
    score_feature = np.concatenate((column_1, column_2, column_3), axis=1)
    return score_feature


if __name__ == '__main__':
    scores = [1, 1, 2, 2, 3, 4, 5]
    score_feature = calculate_score_feature(scores)
    print(score_feature)

