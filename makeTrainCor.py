#coding:utf-8

import random
import os

class candidate_file:

    def __init__(self, filepath, pos_time=1, neu_time=1, neg_time=1):
        self.filepath = filepath
        self.pos_tmie = pos_time
        self.neu_time = neu_time
        self.neg_time = neg_time

    def getFilePath(self):
        return self.filepath

    def getTimes(self):
        return self.pos_tmie, self.neu_time, self.neg_time


def dataAnalysis(filepath):

    print(filepath)
    file = open(filepath,'r')
    lines = file.readlines()

    count = {}
    for i in range(len(lines)/3):

        lab = lines[i*3 +2]
        if count.has_key(lab):
            num = count[lab]
            count[lab] = num + 1
        else:
            count[lab] = 1


    print count



def modifyProportion(candidate_file):
    '''用来调整单个数据集正中负的比例'''
    filepath = candidate_file.getFilePath()
    pos_time, neu_time, neg_time = candidate_file.getTimes()

    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()

    pos_list = []
    neu_list = []
    neg_list = []
    for i in range(len(lines)/3):
        piece = []
        piece.append(lines[i*3])
        piece.append(lines[i * 3 + 1])
        label = lines[i * 3 + 2].replace('\r','')
        piece.append(label)


        if(label == '-1\n'):
            neg_list.append(piece)
        elif(label == '1\n'):
            pos_list.append(piece)
        else:
            piece[2] = '0\n'
            neu_list.append(piece)

    sum_list = []
    sum_list.extend(pos_list * pos_time)
    #print(len(sum_list))
    sum_list.extend(neu_list * neu_time)
    #print(len(sum_list))
    sum_list.extend(neg_list * neg_time)
    #print(len(sum_list))

    return sum_list


def makeTrainCor(file_list, result_dir, filename='target.cor'):

    total_list = []
    for can_file in file_list:
        sum_list = modifyProportion(can_file)
        total_list.extend(sum_list)

    random.shuffle(total_list)

    if(not os.path.exists(result_dir)):  os.makedirs(result_dir)


    length = len(total_list)
    train_index = int(length*0.6)
    dev_index = int(length*0.8)

    train_list = total_list[0 : train_index]
    dev_list = total_list[train_index : dev_index]
    test_list = total_list[dev_index:]





    filepath = result_dir + '/' +filename
    file = open(filepath, 'w')
    for one in total_list:
        for line in one:
            file.write(line)
    file.close()

    filepath = result_dir + '/train.cor'
    file = open(filepath, 'w')
    for one in train_list:
        for line in one:
            file.write(line)
    file.close()

    filepath = result_dir + '/dev.cor'
    file = open(filepath, 'w')
    for one in dev_list:
        for line in one:
            file.write(line)
    file.close()

    filepath = result_dir + '/test.cor'
    file = open(filepath, 'w')
    for one in test_list:
        for line in one:
            file.write(line)
    file.close()



if __name__ == '__main__':

    path1 = 'data_t/train.cor'
    dataAnalysis(path1)
    path2 = 'data_t/test.cor'
    dataAnalysis(path2)

    file_list = []
    file1 = candidate_file(path1, neg_time=10)
    file2 = candidate_file(path2, neu_time=10)
    file_list.append(file1)
    file_list.append(file2)

    result_dir = 'target_models'
    makeTrainCor(file_list, result_dir)

    dataAnalysis(result_dir+'/'+'target.cor')
