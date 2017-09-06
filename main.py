#coding:utf-8

import tensorflow as tf
import argparse
import time
import os
import sys
import random
import gc
import pickle
import util
import time
#from DataManager import DataManager
from NewDataManager import DataManager


FLAGS = None


def train():
  # 装载数据

  print('launch')

  random.seed(FLAGS.seed)
  data = DataManager(FLAGS.dataset)

  wordlist = data.getwordlist()

  #wordlist写出
  if(not os.path.exists(FLAGS.model_dirs)):
    os.makedirs(FLAGS.model_dirs)
  f = open(FLAGS.model_dirs + '/wordlist.bin','wb')
  pickle.dump(wordlist, f)
  f.close()

  x_train,aspect_train,y_train = data.getTrainData()
  x_test, aspect_test, y_test = data.getTestData()

  # 输入数据字数补齐

  max_sentence_length = 30

  train_input = util.dataAlignment(x_train, max_sentence_length)
  test_input = util.dataAlignment(x_test, max_sentence_length)

  # 标签转换为one-hot格式
  train_output = util.dataFormChange(y_train)
  test_output  = util.dataFormChange(y_test)


#-----------------load embed file------------------

  file = open(FLAGS.dataset+'/'+FLAGS.vectorname,'r')
  lines = file.readlines()
  file.close()

  embed_dict = {}#读入训练好的词向量

  for i in range(1, len(lines)):
    line = lines[i].decode('utf-8').strip()
    key = line.split(' ')[0]
    vector = line.split(' ')[1:]
    new_vector = [float(v) for v in vector] # 字符转数字
    embed_dict[key] = new_vector


  #制作embed矩阵
  tmp_dict = {}
  for key in wordlist:
    try:
      vector = embed_dict[key]
      index = wordlist[key]
      tmp_dict[index]=vector
    except:
      index = wordlist[key]
      zerovector = [0.]*100
      tmp_dict[index] = zerovector
  tmp_dict[0] = [0.]*100


  sorted_dict = sorted(tmp_dict.iteritems(), key=lambda d:d[0], reverse = False)
  del tmp_dict
  gc.collect()


  embed_list = [x[1] for x in sorted_dict]

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

  embed_file = open('./results/embed.txt','w')
  for line in embed_list:
    embed_file.write(str(line))
    embed_file.write('\n')

#----------------------搭建graph图--------------------------------

  sess = tf.Session()

  #data are [Batch Size, Sequence Length, Input Dimension]
  # Batch Size 指的是数据有多少行
  with tf.name_scope('input'):
    data = tf.placeholder(tf.int64, [None, max_sentence_length], name='x_input') #原始数据
    target = tf.placeholder(tf.float32, [None, 3], name='y_input')  #原始标签

  with tf.name_scope('embed_vector'):
    embed_tf = tf.placeholder(tf.float32,shape=(None,100),name='embed_tf')  #词向量文件
    vec = tf.nn.embedding_lookup(embed_tf,data)
    tf.summary.histogram("vec", vec)

  with tf.name_scope('lstm'):
    input_keep_prob = tf.placeholder(tf.float32)
    output_keep_prob = tf.placeholder(tf.float32)

    num_hidden = 100 #LSTM隐层数量
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
    val, state = tf.nn.dynamic_rnn(lstm_cell, vec, dtype=tf.float32)
    tf.summary.histogram("val", val)

    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)
    tf.summary.histogram("last", last)

  #softmax
  with tf.name_scope('softmax'):
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]), name='Weight')
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]), name='Bias')
    tf.summary.histogram("weight", weight)
    tf.summary.histogram("bias", bias)
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels= target)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_sum(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)


  #optimizer = tf.train.AdamOptimizer()
  #minimize = optimizer.minimize(cross_entropy)
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.gradient_descent_rate).minimize(cross_entropy)


  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)


  #tensorboard中的graph
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

  #Execution of the graph

  merged = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
  init_op = tf.global_variables_initializer()

  sess.run(init_op)

  batch = FLAGS.batch
  no_of_batches = int(len(train_input)/batch)
  epoch = FLAGS.epoch
  for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
      inp, out = train_input[ptr: ptr+batch], train_output[ptr: ptr+batch]
      ptr += batch
      summary, _, acc, last_print= sess.run([merged, train_step, accuracy,last], {data: inp, target: out, embed_tf:embed_list,input_keep_prob:0.8,output_keep_prob:0.8})

      #for each in last_print: print each

      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (j, acc))
    print "Epoch - ", str(i)

    test_acc, prediction_result = sess.run([accuracy, prediction],
                                             {data: test_input, target: test_output, embed_tf: embed_list,input_keep_prob:1.0,output_keep_prob:1.0})
    print('epoch {:2d} test accuracy {:3.1f}%'.format(i , 100 * test_acc))

  # Create a saver object which will save all the variables
  # 模型保存
  saver = tf.train.Saver()
  saver.save(sess, FLAGS.model_dirs+'/my_test_model',global_step=200)

  acc,prediction_result = sess.run([ accuracy, prediction], {data:test_input, target:test_output, embed_tf:embed_list,input_keep_prob:1.0,output_keep_prob:1.0})

  result_file = open('./results/result.txt','w')
  for line in prediction_result:
    result_file.write(str(line))
    result_file.write('\n')

  print('epoch {:2d} accuracy {:3.1f}%'.format(i + 1, 100 * acc))
  print(prediction_result)
  sess.close()


def main(_):

  time_start = time.time()
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()

  time_end = time.time()
  print(time_end - time_start)

if __name__ == '__main__':

  # 参数解析标准模块
  argv = sys.argv[1:]
  # 模型参数
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, default='lstm')
  parser.add_argument('--seed', type=int, default=int(1000*time.time()))
  parser.add_argument('--dim_hidden', type=int, default=300)
  parser.add_argument('--dim_gram', type=int, default=1)
  parser.add_argument('--optimizer', type=str, default='ADAGRAD')
  parser.add_argument('--gradient_descent_rate', type=int, default=0.00001)
  parser.add_argument('--grained', type=int, default=3)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--lr_word_vector', type=float, default=0.1)
  parser.add_argument('--batch', type=int ,default=1000)
  parser.add_argument('--epoch', type=int, default=500)
  # 文件路径
  parser.add_argument('--dataset', type=str, default='data_zhoulibo', )
  parser.add_argument('--vectorname', type=str, default='used_vector.txt')
  parser.add_argument('--model_dirs', type=str, default='models_zhoulibo')
  # tensorboard数据保存路径
  parser.add_argument('--log_dir', type=str,
                      default='/tmp/tensorflow/lstm/logs/lstm_with_summaries',
                      help='Summaries log directory')

  FLAGS, _ = parser.parse_known_args(argv)

  tf.app.run(main=main, argv=[sys.argv[0]] )
