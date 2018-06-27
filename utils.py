#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np


def read_data(filename): #以读模式打开filename 文件  文件编码是utf8
    with open(filename, encoding="utf-8") as f:
        data = list(f.read()) #将文件以一个大的字符串读进data
                   #将data中的所有字符 一字符为单位 分割成单个字符的list 相当于做了split
    return data


def index_data(sentences, dictionary): #传入array和dict
    shape = sentences.shape
    # print('sentences shape1',shape)
    sentences = sentences.reshape([-1])#拉长为行向量
    # print('sentences shape2',shape)
    index = np.zeros_like(sentences, dtype=np.int32)#创建sentences(是read_data函数返回值)同样shape的0阵
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]#将句子中第i在字在dict中的value也就是对应的编码给index[i]
        except KeyError:
            index[i] = dictionary['UNK']#若sentences[i]这个字不在dict中，也就是说这个字是集合中的低频字出现异常，捕获异常 将UNK的code复值给index[i]
    # for i,j in enumerate(sentences):
    #     index[i] = dictionary.get(j,dictionary['UNK'])
    return index.reshape(shape)#将indexreshape和sentences同样形状，效果是sentences每个字在index中的同样位置对应着这个字的编码


def get_train_data(vocabulary, batch_size, num_steps):  #这里是用课上代码实例中 vocabulary//batch_size 然后再//num_steps 配合yeild
    ##################
    # Your Code here
    ##################
    voca_size = len(vocabulary)
    data_x = vocabulary[:]  #将vocabulary里所有的编码作为输入
    data_y = vocabulary[1:]  #因为要预测下一个字 所以label比如数在同一位置搓后一个字
    data_y.append(vocabulary[-1]) #我的理解是在随机添加一个voca_size 范围内的下表

    part_size = voca_size//batch_size #每次给一个时刻的rnn单元送进去一个字 把文本按照顺序连续的分成batchsize行
    # x_data = np.ndarray(shape=[batch_size,part_size],dtype=np.int32)  #当时设计时一位vocabulary里面时数字结果时汉字所以方式设计的x_data是nint32类型无法装载汉字出错
    # y_data = np.ndarray(shape=[batch_size,part_size],dtype=np.int32)
    x_data = np.ndarray(shape=[batch_size,part_size],dtype=np.unicode)#vocabulary 里面是汉字所以做为容器的xdata只能是字符串类型
    y_data = np.ndarray(shape=[batch_size,part_size],dtype=np.unicode)#numpy的字符转类型有 no.str 应该可以是utf8 np。string是ascii unicode应该是unicode
    for i in range(batch_size):
        x_data[i] = data_x[i*part_size:(i+1)*part_size]
        y_data[i] = data_y[i*part_size:(i+1)*part_size]
    epc = part_size//num_steps  #把每个batchsize行又分成多个numstep个组 便于送个每个rnn单元一个字
    for i in range(epc):
        input_data = x_data[:,i*num_steps:(i+1)*num_steps]
        label_data = y_data[:,i*num_steps:(i+1)*num_steps]
        yield input_data,label_data  #利用生成器返回节省内存







def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))#将words中的字进行统计词频生成(字,词频)的元组list并取前n_words - 1个与['UNK', -1]合并成打的list
    dictionary = dict()
    for word, _ in count:  #遍历count
        dictionary[word] = len(dictionary)#用count里的字生成字典 key是字value是字典中的序号1-nword-1 将低频字置于UNK
    data = list()
    unk_count = 0
    for word in words:  #遍历words
        index = dictionary.get(word, 0)   #生成data列表，它与words项对应 它的每个位置上的元素是words中的元素再dict中的值
        if index == 0:  # dictionary['UNK']#低频单词对应位置data中的值是UNK的索引
            unk_count += 1#统计提品单词个数
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys())) #生成dictionary的翻转字典 即它的value时key key是value
    return data, count, dictionary, reversed_dictionary
#这里的data和上面的index_data实现的是一样的功能？