#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime

import pytz


tz = pytz.timezone('Asia/Shanghai')# 设置时区
current_time = datetime.datetime.now(tz)#记录当前时间


def parse_args(check=True):
    parser = argparse.ArgumentParser()  #创建命令行解析类对象  argparse.ArgumentParser用于命令行解析
    parser.add_argument('--output_dir', type=str, default='./rnn_log', #增加参数 key是output_dir
                        help='path to save log and checkpoint.')

    parser.add_argument('--text', type=str, default='QuanSongCi.txt',#训练文本位置
                        help='path to QuanSongCi.txt')

    parser.add_argument('--num_steps', type=int, default=32, #训练次数
                        help='number of time steps of one sample.')

    parser.add_argument('--batch_size', type=int, default=3,
                        help='batch size to use.')

    parser.add_argument('--dictionary', type=str, default='dictionary.json',  #保存字典jasn文件的路径
                        help='path to dictionary.json.')

    parser.add_argument('--reverse_dictionary', type=str, default='reverse_dictionary.json',#保存字典jasn文件的路径
                        help='path to reverse_dictionary.json.')

    parser.add_argument('--learning_rate', type=float, default=0.001, #学习率
                        help='learning rate')
    parser.add_argument('--embedpath',type=str,default='',help='the path to embeddingfiles.npy')

    FLAGS, unparsed = parser.parse_known_args() #将参数生成字典在FLAGS便于调用

    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    for x in dir(FLAGS):   
        print(getattr(FLAGS, x))  #getattr获取静态属性制定对象的制定属性值 可以设置默认值 还有setattr 和hasattr 结果应该是true
