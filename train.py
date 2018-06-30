#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import numpy as np
import tensorflow as tf
import utils
from model import Model
from utils import read_data
from utils import build_dataset
from flags import parse_args
FLAGS, unparsed = parse_args()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


vocabulary = read_data(FLAGS.text) #读入text并以字为单位形成list
print('Data size', len(vocabulary))

data1, count1, dictionary1, reversed_dictionary1 = build_dataset(vocabulary, 5000)#调用utils里的build_dataset生成字典的另一种方法
with open(FLAGS.dictionary, 'r',encoding='utf8') as inf:#载入字典
    dictionary = json.load(inf)

with open(FLAGS.reverse_dictionary,'r',encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf,encoding='utf-8')


model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)

model.build()


with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph) #set checkpoint dir

    saver = tf.train.Saver(max_to_keep=5)#保存检查点
    sess.run(tf.global_variables_initializer())#初始化全局变量
    sess.run(tf.local_variables_initializer())#初始化本地临时变量
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir) #获取最近检查点文件
        saver.restore(sess, checkpoint_path)#恢复检查点
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')

    for x in range(1):
        logging.debug('epoch [{0}]....'.format(x))
        state = sess.run(model.state_tensor)
        for dl in utils.get_train_data(vocabulary, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):
            X = []
            for i in range(len(dl)):  #是一个 batch_size* num_steps2d矩阵 get_train_data返回的是两组值
                h,w = dl[i].shape
                temp = np.ndarray(shape=[h,w],dtype=np.int32)
                for j in range(h):
                    temp[j] = [dictionary.get(w,0) for w in dl[i][j]] #将矩阵里的每一个文字都变成他字典中所对应的valu
                X.append(temp)
            # X = utils.index_data(dl[0],dictionary)
            # Y = utils.index_data(dl[1],dictionary)
            feed_dict = {model.X:X[0],                         #构造feed字典  dl[0]是input dl[1]是output
                         model.Y:X[1],
                         model.state_tensor:state,
                         model.is_training:1,     # 训练时维珍
                         model.keep_prob:0.8}      #0.8
            gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.outputs_state_tensor, model.loss, model.merged_summary_op], feed_dict=feed_dict)
            summary_string_writer.add_summary(summary_string, gs) #

            if gs % 10 == 0:
                logging.debug('step [{0}] loss [{1}]'.format(gs, l))            # 每隔10步输出步数和loss
                save_path = saver.save(sess, os.path.join(                    #每隔10步保存一次检查点
                    FLAGS.output_dir, "model.ckpt"), global_step=gs)
    summary_string_writer.close()
