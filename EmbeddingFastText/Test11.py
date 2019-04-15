#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
Created on 4/8/2019 16:12 

如果不考虑标签的层级关系，直接分类二级或者三级标签，需要解决两个问题

    1. 部分标签的名称是相同，所以不同级别的标签的表示方式变为：
        label   本地生活--游戏充值--游戏点卡
        level = 1   return 本地生活
        level = 2   return 本地生活--游戏充值
        level = 3   return 本地生活--游戏充值--游戏点卡

    2. 部分标签，尤其是部分三级标签，对应的数据量太少，
        在切分数据集的时候，很可能导致该标签下所有的数据都在训练集的范围内，或都在测试集的范围内。
        因此需要对每一个标签对应的数据打乱后切分训练集和测试集，
        再将各个标签的训练集和测试集聚类，并再次打乱，
        而后的分词等操作不变，具体见下方数据预测处理流程

此外，这次的代码中增加了方法 get_type_dict ，它可以根据特定的级别，获得一个功能强悍的字典，
每个标签都对应一个列表，内容包括 标签对应数据的开始索引，标签对应数据的结束索引，标签的种类索引（用于分类），
详见 get_type_dict 的注释说明。


数据预处理流程

    获得源数据 ori_data
    指定 level 的源数据的标签字典 type_dict
    for type in type_dict
        获得 type 对应的数据 data
        打乱 data
        对 data 按照 perc 切割成 train 和 test
    对 for 循环中得到的所有 train 和 test 聚类，得到 trains 和 tests
    将 trains 和 tests 打乱
    将 trains 切分为 tra_sams 和 tra_labs ，tests 切分为 tes_sams 和 tes_labs
    对 tra_sams 和 tes_sams 进行分词
    将 tra_labs 和 tes_labs 转为索引

@author: SmacUL 
"""

import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import Input
from keras import utils


def create_DNN_model(emb_size, cate, tok_dim=64):
    """ 全连接模型

    :param emb_size:    词典的大小，即 count.txt 文件的词表长度
    :param cate:        分类数
    :param tok_dim:     词特征维度
    :return:
    """
    print("正在使用 DNN 模型")
    text_input = Input(shape=(None,), dtype='int32', name='sam')
    emb = layers.Embedding(emb_size, tok_dim, input_length=max_word_num)(text_input)
    flat = layers.Flatten()(emb)
    dnn1 = layers.Dense(256, activation='relu')(flat)
    dnn2 = layers.Dense(64, activation="relu")(dnn1)
    classifier = layers.Dense(cate, activation='softmax')(dnn2)
    model = models.Model(text_input, classifier)
    print(model.summary())
    return model


def create_CNN_model(emb_size, cate, tok_dim=64):
    """ CNN 模型

    :param emb_size:    词典的大小，即 count.txt 文件的词表长度
    :param cate:        分类数
    :param tok_dim:     词特征维度
    :return:
    """
    print("正在使用 CNN 模型")
    text_input = Input(shape=(None,), dtype='int32', name='sam')
    emb = layers.Embedding(emb_size, tok_dim, input_length=max_word_num)(text_input)
    cnn1 = layers.Convolution1D(32, 3, activation='relu', padding='same', strides=1)(emb)
    cnn2 = layers.Convolution1D(32, 3, activation='relu', padding='same', strides=1)(cnn1)

    flat = layers.Flatten()(cnn2)
    dnn1 = layers.Dense(128, activation='relu')(flat)
    drop = layers.Dropout(0.5)(dnn1)
    classifier = layers.Dense(cate, activation='softmax')(drop)
    model = models.Model(text_input, classifier)
    print(model.summary())
    return model


def create_Text_CNN_model(emb_size, cate, tok_dim=64):
    """ Text_CNN 模型

    Embedding 层下是三个卷积核大小分别为 3 4 5 的 CNN 和 MaxPooling 分支层，
    再将三个分支合并，展平，使用全连接拟合，分类。

    :param emb_size:    词典的大小，即 count.txt 文件的词表长度
    :param cate:        分类数
    :param tok_dim:     词特征维度
    :return:
    """
    print("正在使用 Text_CNN 模型")
    text_input = Input(shape=(None,), dtype='int32', name='text')
    emb = layers.Embedding(emb_size, tok_dim, input_length=max_word_num)(text_input)

    cnn1 = layers.Convolution1D(16, 3, activation='relu', padding='same', strides=1)(emb)
    cnn2 = layers.Convolution1D(16, 4, activation='relu', padding='same', strides=1)(emb)
    cnn3 = layers.Convolution1D(16, 5, activation='relu', padding='same', strides=1)(emb)

    mx1 = layers.MaxPool1D(pool_size=2)(cnn1)
    mx2 = layers.MaxPool1D(pool_size=2)(cnn2)
    mx3 = layers.MaxPool1D(pool_size=2)(cnn3)

    cnn_all = layers.concatenate([mx1, mx2, mx3], axis=-1)
    flat = layers.Flatten()(cnn_all)
    dnn1 = layers.Dense(128, activation='relu')(flat)
    classifier = layers.Dense(cate, activation='softmax')(dnn1)
    model = models.Model(text_input, classifier)
    print(model.summary())
    return model


def create_Fast_Text_model(emb_size, cate, tok_dim=64):
    """ Fast_Text 的 Keras 简单实现版本

    Embedding 层下是三个卷积核大小分别为 3 4 5 的 CNN 和 MaxPooling 分支层，
    再将三个分支合并，展平，使用全连接拟合，分类。

    :param emb_size:    词典的大小，即 count.txt 文件的词表长度
    :param cate:        分类数
    :param tok_dim:     词特征维度
    :return:
    """
    print("正在使用 Fast_Text 模型")
    text_input = Input(shape=(None,), dtype='int32', name='text')
    emb = layers.Embedding(emb_size, tok_dim, input_length=max_word_num)(text_input)
    gap = layers.GlobalAveragePooling1D()(emb)
    classifier = layers.Dense(cate, activation='softmax')(gap)
    model = models.Model(text_input, classifier)
    print(model.summary())
    return model


def get_ori_data(path, encode='gb18030'):
    """ 获取源数据

    数据总数量 499999 条

    :param path:    数据路径
    :param encode:  源数据文件编码，即 Train.tsv 的文件编码
    :return:        源数据，nparray 类型
    """
    data = pd.read_csv(path, sep='\t', encoding=encode, nrows=None)
    print("成功获取数据！")
    return np.array(data)


def get_level_label(label, level):
    """ 组装标签

    示例：
        label   本地生活--游戏充值--游戏点卡
        level = 1   return 本地生活
        level = 2   return 本地生活--游戏充值
        level = 3   return 本地生活--游戏充值--游戏点卡

    :param label:   单条完成的标签
    :param level:   需要获得的标签的级别
    :return:        重组后的标签
    """
    if level == 1:
        return label.split("--")[0]
    elif level == 3:
        return label
    else:
        return label.split("--")[0] + "--" + label.split("--")[1]


def get_type_dict(data, level):
    """ 获取指定级别的类别标签

    字典格式
        {'本地生活--游戏充值': [0, 349, 0], ··· }
        {标签 -- [开始下标（闭区间） , 结束下标（开区间） , 索引值], ··· }

    :param data:    按照顺序的排列的源数据
    :param level:   标签的级别，最小为 1 ，最大为 3
    :return:        类型字典 {'本地生活--游戏充值': [0, 350, 0], ··· }
                    标签 -- [开始下标（闭区间） , 结束下标（开区间） , 索引值]


    算法说明
        # length 是一个下标指针，初始指向数据开头，代表还未遍历过的开始数据的下标
        while
            # 获取 length 所指的 label La
            La = get_label(length)
            # 每个 for 循环将获得一类标签的数据起始位置和索引值
            # start 指向当前 for 循环的开始下标位置，在循环中获取 end 的位置
            for i from start to len(data)
                # 获得当前的下标所指的 label Lb
                Lb = get_label(i)
                if La != Lb
                    # 说明本类标签遍历完成
                    # 获取 end
                    set(end) = i - 1
                    # 刷新 length ，指向下一类标签的开头位置
                    set(length) = i
                    break
            if end == start - 1
                # 说明 end 没有被修改过，在上面的 for 循环中没有获得 Lb ， 即到达了 data 的末尾
                # 设置 end 为 len(data) - 1
                set(end) = len(data) - 1
                # 设置 length 为 len(data) ，使算法跳出 while 循环
                set(length) = len(data)
            set type_dict
    """
    print("当前开始获取 %d 级标签的类别字典" % level)
    type_dict = {}
    length = 0
    data_length = len(data)
    index = -1
    while length < data_length:
        start_label = get_level_label(data[length, 1], level)
        start = length
        end = length - 1
        index += 1
        for i in range(start, data_length):
            cur_label = get_level_label(data[i, 1], level)
            if cur_label != start_label:
                length = i
                end = i - 1
                break
        if end == start - 1:
            length = data_length
            end = length - 1
        type_dict[start_label] = [start, end + 1, index]
    print(type_dict)
    print("类别字典总长度 % d" % len(type_dict))
    return type_dict


def divide_sentences_to_tokens(data):
    """ 将商品描述信息，即样本，转化为分词数据

    使用 Jieba 分词

    :param data:    商品描述信息，类型为 nparray ，可以是训练样本集 tra_sams 或测试样本集 tes_labs
    :return:        data 对应的分词数据，类型为 nparray
    """
    print("数据分词开始！")
    toks_all = []
    for da in data:
        toks = jieba.cut(da)
        toks_all.append(toks)
    print("数据分词结束！")
    return np.array(toks_all)


def create_tag(path):
    """ 依据 count.txt 文件生成词典

    词典的每一项包括分词及其对应的索引值，
    词典长度为 count.txt 文件的词汇数。

    :param path:    count.txt 文件的路径
    :return:        词典
    """
    print("开始获取分词索引字典！")
    dict = {}
    fp = open(path, 'r', encoding="utf8")
    ls = fp.readlines()
    for i, item in enumerate(ls):
        dict[item.split(" ")[0]] = i
    print("分词索引字典获取结束！")
    return dict


def digitize_tokens(sams, dict, max_word_num=45):
    """ 将分词转为数字索引

    每条商品信息包含的分词数都不相同，词典也不可能将所有分词包括在内，
    如果分词空缺，或是在词典中不存在，分词索引，都将被设置为 0

    :param sams:            分词后的商品描述信息
    :param dict:            词典，由 create_tag 方法生成
    :param max_word_num:    每条商品描述信息中，分词的最大数量
                            此处可以有更好的办法，Keras 提供了相应的 API ，可以自动设置分词的最大数量。
    :return:                转为数字索引的分词数据
    """
    print("正在将分词转为数字索引")
    toks_index = np.zeros([len(sams), max_word_num], dtype=int)
    for i, toks in enumerate(sams):
        for j, tok in enumerate(toks):
            if dict.__contains__(tok):
                toks_index[i][j] = dict[tok]
    print("索引转化完毕")
    return toks_index


def digitize_label(labs, type_dict, level):
    """ 将标签转为数字索引

    数字索引集需要再用 keras.utils.to_categorical 方法包装一遍

    :param labs:        标签集合，可以是 tra_labs 或 tes_labs
    :param type_dict:   种类词典
    :param level:       标签级别
    :return:
    """
    y = []
    for lab in labs:
        y.append(type_dict[get_level_label(lab, level)][2])
    return utils.to_categorical(np.array(y))


def draw_result(history):
    """ 绘制出学习的结果图

    :param history:     模型训练结果
    :return:
    """
    plt.plot(range(1, len(history.history['acc']) + 1), history.history['acc'], 'bo', label='train_acc')
    plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], 'b', label='train_loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 读取数据
    ori_data = get_ori_data("./train.tsv")
    # 分类级别
    level = 3
    # 获取类别字典
    type_dict = get_type_dict(ori_data, level=level)

    # 切分比例
    perc = 0.8
    # 将每一类打乱后切分成训练集和测试集，并将训练集和测试集归类
    trains = []
    tests = []
    for i, tp in enumerate(type_dict):
        infos = type_dict[tp]
        cur_data = ori_data[infos[0]: infos[1]]
        np.random.shuffle(cur_data)
        train = cur_data[0: int(perc * len(cur_data))]
        test = cur_data[int(perc * len(cur_data)): len(cur_data)]
        trains += list(train)
        tests += list(test)
    # 将 trains 和 tests 转为 nparray 类型，分别再次打乱
    trains = np.array(trains)
    tests = np.array(tests)
    np.random.shuffle(trains)
    np.random.shuffle(tests)

    # 切分四个数据集
    tra_sams = trains[:, 0]
    tes_sams = tests[:, 0]
    tra_labs = trains[:, 1]
    tes_labs = tests[:, 1]

    # 分词
    tra_toks = divide_sentences_to_tokens(tra_sams)
    tes_toks = divide_sentences_to_tokens(tes_sams)

    # 商品描述信息中的最大分词数量
    max_word_num = 70

    # 获取分词索引字典
    index_dict = create_tag("./count.txt")
    tra_toks_index = digitize_tokens(tra_toks, index_dict, max_word_num)
    tes_toks_index = digitize_tokens(tes_toks, index_dict, max_word_num)

    # 标签数字化
    tra_y = digitize_label(tra_labs, type_dict, level=level)
    tes_y = digitize_label(tes_labs, type_dict, level=level)

    print("------------------------------------数据预处理结束---------------------------------------")

    tok_dim = 64
    epochs = 5
    batch_size = 1024

    model = create_CNN_model(len(index_dict), len(type_dict), tok_dim=tok_dim)
    # model = create_DNN_model(len(index_dict), len(type_dict), tok_dim=tok_dim)
    # model = create_Text_CNN_model(len(index_dict), len(type_dict), tok_dim=tok_dim)
    # model = create_Fast_Text_model(len(index_dict), len(type_dict), tok_dim=tok_dim)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(tra_toks_index, tra_y, epochs=epochs, batch_size=batch_size)
    tes_loss, tes_acc = model.evaluate(tra_toks_index, tra_y, batch_size=batch_size)
    print("test loss is : ", tes_loss)
    print("test accuracy is : ", tes_acc)
    # 绘制模型训练结果图
    draw_result(history)
