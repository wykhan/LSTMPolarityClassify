# coding:utf-8

import tensorflow as tf
import argparse
import time
import sys
import random
import gc
import pickle
import os
import util
from NewDataManager import DataLoader

FLAGS = None


def result_analysis( raw_content_test, y_test, result, filepath):
    sum_list = []

    neg_count = 0
    pos_count = 0

    for i in range(0, len(y_test)):
        predict_class = str(list(result[i]).index(max(result[i])) -1)

        #统计正负预测的个数
        if(predict_class == '1'): pos_count+=1
        elif(predict_class == '-1'): neg_count+=1



        if(predict_class == str(y_test[i])):
            flag = True
        else:
            flag = False

        newline = '\t'.join([str(flag), str(y_test[i]), predict_class, str(result[i]), raw_content_test[i]])+'\n'
        sum_list.append(newline.encode('utf-8'))
    file = open(filepath,'w')
    file.writelines(sum_list)
    file.close()

    file=open(filepath,'r')
    lines = file.readlines()
    matrix = [[0,0,0],[0,0,0],[0,0,0]]

    for line in lines:
        i = int(line.split('\t')[1]) + 1
        j = int(line.split('\t')[2]) + 1
        if i == 3: i=1
        try:
            count = matrix[i][j]
            matrix[i][j] = count + 1
        except:
            print 'lab wrong'



    print matrix[0]
    print matrix[1]
    print matrix[2]

    score = float(pos_count)/(pos_count + neg_count)
    print('好评得分为：%0.2f%%'%(score*100))

    file = open(filepath, 'a')
    file.write('\t'.join([str(x) for x in matrix[0]])+'\n')
    file.write('\t'.join([str(x) for x in matrix[1]])+'\n')
    file.write('\t'.join([str(x) for x in matrix[2]])+'\n')
    file.write('好评得分为：%0.2f%%'%(score*100))

def predict():
    # 装载数据

    print('launch')
    random.seed(FLAGS.seed)

    #载入wordlist
    if(os.path.exists(FLAGS.model_dirs+'/'+FLAGS.wordlist)):
        f = open(FLAGS.model_dirs+'/'+FLAGS.wordlist,'rb')
        wordlist = pickle.load(f)
    else:
        raise NameError

    # 载入测试数据
    dataloader = DataLoader(FLAGS.testfile, wordlist)
    raw_content_test, x_test, aspect_test, y_test = dataloader.getData()

    # 输入数据字数补齐
    max_sentence_length = 30
    test_input = util.dataAlignment(x_test, max_sentence_length)

    # 标签转换为one-hot格式
    test_output = util.dataFormChange(y_test)

    # -----------------load embed file------------------

    file = open(FLAGS.model_dirs + '/' + FLAGS.vectorname, 'r')
    lines = file.readlines()
    file.close()

    embed_dict = {}  # 读入训练好的词向量

    for i in range(1, len(lines)):
        line = lines[i].decode('utf-8').strip()
        key = line.split(' ')[0]
        vector = line.split(' ')[1:]
        new_vector = [float(v) for v in vector]  # 字符转数字
        embed_dict[key] = new_vector

    # 制作embed矩阵
    tmp_dict = {}
    for key in wordlist:
        try:
            vector = embed_dict[key]
            index = wordlist[key]
            tmp_dict[index] = vector
        except:
            index = wordlist[key]
            zerovector = [0.] * 100
            tmp_dict[index] = zerovector
    tmp_dict[0] = [0.] * 100

    sorted_dict = sorted(tmp_dict.iteritems(), key=lambda d: d[0], reverse=False)
    del tmp_dict
    gc.collect()

    embed_list = [x[1] for x in sorted_dict]

    # embed_list = np.random.rand(12158, 100)

    zero_count = 0
    unzero_count = 0
    for embed in embed_list:
        total = sum(embed)
        if (total == 0.0):
            zero_count += 1
        else:
            unzero_count += 1

    print(zero_count)
    print(unzero_count)

    embed_file = open('./results/embed.txt', 'w')
    for line in embed_list:
        embed_file.write(str(line))
        embed_file.write('\n')

    # ----------------------搭建graph图--------------------------------

    sess = tf.Session()

    # data are [Batch Size, Sequence Length, Input Dimension]
    # Batch Size 指的是数据有多少行
    with tf.name_scope('input'):
        data = tf.placeholder(tf.int64, [None, max_sentence_length], name='x_input')  # 原始数据
        target = tf.placeholder(tf.float32, [None, 3], name='y_input')  # 原始标签

    with tf.name_scope('embed_vector'):
        embed_tf = tf.placeholder(tf.float32, shape=(None, 100), name='embed_tf')  # 词向量文件
        vec = tf.nn.embedding_lookup(embed_tf, data)
        tf.summary.histogram("vec", vec)

    with tf.name_scope('lstm'):
        input_keep_prob = tf.placeholder(tf.float32)
        output_keep_prob = tf.placeholder(tf.float32)

        num_hidden = 100  # LSTM隐层数量
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=input_keep_prob,
                                                  output_keep_prob=output_keep_prob)
        val, state = tf.nn.dynamic_rnn(lstm_cell, vec, dtype=tf.float32)
        tf.summary.histogram("val", val)

        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        tf.summary.histogram("last", last)

    # softmax
    with tf.name_scope('softmax'):
        weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]), name='Weight')
        bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]), name='Bias')
        tf.summary.histogram("weight", weight)
        tf.summary.histogram("bias", bias)
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_sum(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # optimizer = tf.train.AdamOptimizer()
    # minimize = optimizer.minimize(cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


    # Execution of the graph
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)

    # 模型载入
    # saver = tf.train.import_meta_graph(FLAGS.model_dirs+'/'+FLAGS.model_load)
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dirs))

    acc, prediction_result = sess.run([accuracy, prediction],
                                      {data: test_input, target: test_output, embed_tf: embed_list,
                                       input_keep_prob: 1.0, output_keep_prob: 1.0})


    result_file = open('./results/result.txt', 'w')
    for line in prediction_result:
        result_file.write(str(line))
        result_file.write('\n')

    print('epoch {:2d} accuracy {:3.1f}%'.format(i + 1, 100 * acc))
    print(prediction_result)
    sess.close()

    result_analysis(raw_content_test, y_test, prediction_result, './results/sum.txt')

def main(_):

    predict()


if __name__ == '__main__':

    # 参数解析标准模块
    argv = sys.argv[1:]
    # 模型参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='lstm')
    parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--dim_gram', type=int, default=1)

    # 模型文件路径
    parser.add_argument('--model_dirs', type=str, default='models_zhoulibo')
    parser.add_argument('--vectorname', type=str, default='used_vector.txt')
    parser.add_argument('--model_load', type=str, default='my_test_model-200.meta')
    parser.add_argument('--wordlist', type=str, default='wordlist.bin')
    #测试文件路径
    parser.add_argument('--testfile', type=str, default='data_zhoulibo/test.cor')
    #parser.add_argument('--testfile', type=str, default='吴亦凡.txt')

    FLAGS, _ = parser.parse_known_args(argv)

    tf.app.run(main=main, argv=[sys.argv[0]])
