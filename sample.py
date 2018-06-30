#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import numpy as np
import tensorflow as tf
import datetime
import utils
from model import Model

from flags import parse_args
FLAGS, unparsed = parse_args()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8') #载入dictionary

with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf: #载入 reverse_dictionary
    reverse_dictionary = json.load(inf, encoding='utf-8')


reverse_list = [reverse_dictionary[str(i)] 
                for i in range(len(reverse_dictionary))]
titles = ['江神子', '蝶恋花', '渔家傲']

#Model在model。py
model = Model(learning_rate=FLAGS.learning_rate, batch_size=1, num_steps=1)
model.build()

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
        state = sess.run(model.state_tensor) 
        # feed title
        for head in title:#遍历词组的每一个字
            input = utils.index_data(np.array([[head]]), dictionary)

            feed_dict = {model.X: input,
                         model.state_tensor: state,
                         model.keep_prob: 1.0,
                         model.is_training:0}

            pred, state = sess.run(                            
                [model.predictions, model.outputs_state_tensor], feed_dict=feed_dict)

        sentence = title  #SENTENCE自加

        word_index = np.random.choice(range(len(reverse_list)),p=pred[0])
        # generate sample
        for i in range(64):  #执行64遍
            feed_dict = {model.X: [[word_index]],#将返回的索引值继续作为输入预测下一个字是什么
                         model.state_tensor: state, #将0初始状态赋值给init stale
                         model.keep_prob: 1.0,
                         model.is_training:0}

            pred, state = sess.run(
                [model.predictions, model.outputs_state_tensor], feed_dict=feed_dict) 

            # word_index = pred[0] .argsort()[-1] 
            p_index = pred[0]
            # word = np.take(reverse_list, word_index)
            word_index = np.random.choice(range(len(reverse_list)),p=p_index)
            word = reverse_dictionary[str(word_index)]
            sentence = sentence + word 
        logging.debug('==============[{0}]=============='.format(title))
        logging.debug(sentence)
        # s_dir = os.path.dirname(FLAGS.output_dir)
        # s_name = str(datetime.datetime.now())
        # with open(os.path.join(s_dir,s_name),'w',encoding='utf8',errors='ignore') as fig:
        #     fig.write(sentence)
