#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from flags import parse_args
FLAGS,unparse = parse_args()


class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float
            学习率.
        batch_size : int
            batch_size.
        num_steps : int
            RNN有多少个time step，也就是输入数据的长度是多少.
        num_words : int
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int
            embding中，编码后的字向量的维度 也就是隐层单元的个数
        rnn_layers : int
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起
            
        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')#训练多少步就有多少列
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob',)
        self.is_training = tf.placeholder(tf.int32,name='is_training',)
        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                # print(embedding_file)
                embedding = np.load(embedding_file)#np.load是将npy文件保存的array载入内存形成一个array 参数是npy文件路径 而np.save是将内存中的array保存在文件中
                embed = tf.constant(embedding, name='embedding')#生成常量
            else:
                # if not, initialize an embedding and train it.
                print('no emdedding')
                embed = tf.get_variable(                  #获取embedding 输入层到隐层的权重矩阵
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            data = tf.nn.embedding_lookup(embed, self.X)#input和权重矩阵查表得到embeding后的输入 batch_size*num_steps*dim_embedding

        with tf.variable_scope('rnn'):
            '''MY CODE HERE'''
            #定义 cell unit
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dim_embedding,forget_bias=1.0,state_is_tuple=False)
            
            if self.is_training == 1 and self.keep_prob < 1:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
              
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for i in range(self.rnn_layers)],state_is_tuple=False)
            
            self.state_tensor = lstm_cell.zero_state(self.batch_size,dtype=tf.float32) #自动转换成 batch*dim_embedding
            init_state = self.state_tensor
           
            if self.is_training == 1 and self.keep_prob < 1:
                data = tf.nn.dropout(data,keep_prob=self.keep_prob)
            #运次rnn 使用f.nn.dynamic_rnn
            '''
             We initialize the hidden states to zero. We then use the
            final hidden states of the current minibatch as the initial hidden state of the subsequent minibatch
            (successive minibatches sequentially traverse the training set).
            '''
            seq_output,final_state = tf.nn.dynamic_rnn(lstm_cell,data,initial_state=init_state) #将上一个ba
            self.outputs_state_tensor = final_state
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])  # [batchsize*timesteps ，dim_embedding]



        with tf.variable_scope('softmax'):
            w = tf.get_variable('w',[self.dim_embedding,self.num_words],initializer=tf.truncated_normal_initializer(stddev=0.001))#没初始化
            b = tf.get_variable('b',[self.num_words,],initializer=tf.constant_initializer(0))
            logits = tf.matmul(seq_output_final,w)+b  #batchsize*timesteps ，num_words
            # logits = tf.reshape(logits,[self.batch_size,self.num_steps,self.num_words])
        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.Y, [-1]),logits=logits)#yreshape后是batchsize*timesteps列1行logits是batchsize*timesteps ，num_words 而groundtruth中每一个字都对应着numwords个概率值
        mean, var = tf.nn.moments(logits, -1)#在最后words这个个维度上求logist的均值方差
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('logits_loss', self.loss)

        var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(var))
        tf.summary.scalar('var_loss', var_loss)
        # 把标准差作为loss添加到最终的loss里面，避免网络每次输出的语句都是机械的重复
        self.loss = self.loss + var_loss
        tf.summary.scalar('total_loss', self.loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()
