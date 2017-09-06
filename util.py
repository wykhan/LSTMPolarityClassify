#coding:utf-8
import numpy as np


def dataFormChange(y):
    '''
    转化成onehot结构
    '''
    new_y = np.zeros((len(y), 3))
    for i in range(len(y)):
        label_value = y[i]
        if label_value >= 2: label_value = 0
        new_y[i][label_value + 1] = 1
    return new_y


def dataAlignment(dataset, max_length):
    '''
    将不同长度的语料补齐为相同长度
    '''
    data_with_same_length = []
    for i in range(len(dataset)):
        newline = [0] * max_length
        for j in range(min(len(dataset[i]), max_length)):
            newline[j] = dataset[i][j]
        newarray = np.array(newline)
        data_with_same_length.append(newarray)
    return data_with_same_length
