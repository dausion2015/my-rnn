#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging

import numpy as np
import tensorflow as tf

import utils
from model import Model

from flags import parse_args
FLAGS, unparsed = parse_args()
# embedding_file_path = FLAGS.embeddingflies
# embedding_file = np.load(embedding_file_path)#难道他这个loads和load的区别也和json一样 不对jspn都是先打开文件句柄 只是read是否的问题

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

# is_training = False
# print('in sample training is',is_training)
with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8') #载入dictionary

with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf: #载入 reverse_dictionary
    reverse_dictionary = json.load(inf, encoding='utf-8')


reverse_list = [reverse_dictionary[str(i)] #高频字list取出所有的valuej即字典中的所有字放入list 相当于reverse_dictionary.values
                for i in range(len(reverse_dictionary))]
titles = ['江神子', '蝶恋花', '渔家傲']

#Model在model。py
model = Model(learning_rate=FLAGS.learning_rate, batch_size=1, num_steps=1)
model.build(embedding_file = FLAGS.embeddingflies)

with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)#将op点写入文件output_dir

    saver = tf.train.Saver(max_to_keep=5) #保存检查点
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)#获取最近检查点文
        saver.restore(sess, checkpoint_path)#恢复检查点
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')
        exit(0)

    for title in titles: #遍历titles每一个词组
        state = sess.run(model.state_tensor) #运行0初始化init state  值存于state
        # feed title
        for head in title:#遍历词组的每一个字
            input = utils.index_data(np.array([[head]]), dictionary)#返回字对应的编码以同样shape返回返回一个字的编码但是是二维矩阵返回在dictionary中head对应的value

            feed_dict = {model.X: input,
                         model.state_tensor: state,
                         model.keep_prob: 1.0,
                         model.is_training:0}

            pred, state = sess.run(                            #计算[model.predictions, model.outputs_state_tensor洗一个循环时这里的state又被送到feed里
                [model.predictions, model.outputs_state_tensor], feed_dict=feed_dict)

        sentence = title  #SENTENCE自加

        word_index = pred[0].argsort()[-1]#这里的pred是最后一个head的pred 预测下一个自codepred是batch*num_steps,num_words2维 把pre里概率最大值的在高频字的索引返回存于word_index

        # generate sample
        for i in range(64):  #执行64遍
            feed_dict = {model.X: [[word_index]],#将返回的索引值继续作为输入预测下一个字是什么
                         model.state_tensor: state, #将0初始状态赋值给init stale
                         model.keep_prob: 1.0,
                         model.is_training:0}

            pred, state = sess.run(
                [model.predictions, model.outputs_state_tensor], feed_dict=feed_dict) #第一次循环以后这里的state都是下一次feed里的state计算predictions, finalstate

            word_index = pred[0] .argsort()[-1] #返回最大概率之的索引号
            word = np.take(reverse_list, word_index) #在高频字list中挑出word_index对应的字 np.take将word_index中的元素作为索引查找(reverse_list中的元素返回
            sentence = sentence + word #把所有预测的64字都累加到sentence

        logging.debug('==============[{0}]=============='.format(title))
        logging.debug(sentence)
