#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os


from flags import parse_args


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd())) #获取当前工作dir
    w_d = os.path.dirname(os.path.abspath(__file__)) #脚本所在dir
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d) #os。chdir是改变目录

    cmd = ""
    for parm in ["output_dir", "text", "num_steps", "batch_size", "dictionary", "reverse_dictionary", "learning_rate",'embedpath']:
        try:
            cmd += ' --{0}={1}'.format(parm, getattr(FLAGS, parm))#拼接命令字符串 通过过getatr获取外部传入的参数值 FLAGS是外部命令参数字典
        except:
            raise Exception('error')
        # print('**************************************************************',getattr(FLAGS, parm))
    for i in range(30):
       # train 1 epoch
        print('################    train    ################')

        # print('is_training is',FLAGS.)
        p = os.popen('python ./train.py' + cmd)#通过popen运行train。py拼接好的命令popen能以传入字符换参数形式运行系统命令并且能获取返回结果显示

        for l in p: #获取返回文本结果
            print(l.strip()) #去掉文本前后空格

        # eval
        print('################    eval    ################')

        p = os.popen('python ./sample.py' + cmd)#通过popen运行sample.py拼接好的命令
        for l in p:
            print(l.strip())

