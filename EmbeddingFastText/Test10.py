#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
Created on 3/24/2019 10:11 

[major reference](https://zhuanlan.zhihu.com/p/29201491)

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


def divide_data(data, perc=0.8):
    """ 切分源数据为四个数据集，见 return

    :param data:    源数据，要求为 nparray 类型
    :param perc:    训练集的比例，默认为 0.8
    :return:        tra_sams 训练样本集，tes_sams 测试样本集，tra_labs 训练标签集，tes_labs 测试标签集
    """
    print("数据切分开始！")
    sams = data[:, 0]
    labs = data[:, 1]

    tra_sams = sams[0: int(perc * len(sams))]
    tra_labs = labs[0: int(perc * len(labs))]
    tes_sams = sams[int(perc * len(sams)): len(sams)]
    tes_labs = labs[int(perc * len(labs)): len(labs)]
    print("数据切分结束！")
    return tra_sams, tes_sams, tra_labs, tes_labs


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
    toks_index = np.zeros([len(sams), max_word_num], dtype=int)
    for i, toks in enumerate(sams):
        for j, tok in enumerate(toks):
            if dict.__contains__(tok):
                toks_index[i][j] = dict[tok]
    return toks_index


def digitize_label(labs, type_dict):
    """ 将标签转为数字索引

    数字索引集需要再用 keras.utils.to_categorical 方法包装一遍

    :param labs:        标签集合，可以是 tra_labs 或 tes_labs
    :param type_dict:   种类词典，一分类一共 22 种
    :return:
    """
    y = []
    for lab in labs:
        y.append(type_dict[lab.split('--')[0]])
    return utils.to_categorical(np.array(y))


def choose_model(name, emb_size, cate, tok_dim=64):
    """ 选择模型

    相当于一个工厂方法，
    可选 全连接 DNN ；卷积神经网络 CNN ；文本卷积神经网络 Text_CNN ；Keras 版本的 fastText Fast_Text

    :param name:        模型名称
    :param emb_size:    词嵌入层中词典的大小，即 count.txt 的词汇条数
    :param cate:        分类数量
    :param tok_dim:     分词的特征维度
    :return:            相应的模型构建方法
    """
    if name == "DNN":
        return create_DNN_model(emb_size, cate, tok_dim=tok_dim)
    elif name == "CNN":
        return create_CNN_model(emb_size, cate, tok_dim=tok_dim)
    elif name == "Text_CNN":
        return create_Text_CNN_model(emb_size, cate, tok_dim=tok_dim)
    elif name == "Fast_Text":
        return create_Fast_Text_model(emb_size, cate, tok_dim=tok_dim)
    else:
        print("没有此模型！")


def model_configure(model):
    """ 模型配置

    :param model:
    :return:
    """
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


def train_model(model, x, y, epochs=20, batch_size=64):
    """ 模型训练

    :param model:
    :param x:
    :param y:
    :param epochs:
    :param batch_size:
    :return:
    """
    return model.fit(x, y, epochs=epochs, batch_size=batch_size)


def model_evaluate(model, x, y, batch_size=64):
    """ 评估模型

    :param model:
    :param x:
    :param y:
    :param batch_size:
    :return:            测试集的损失和测试集的正确率
    """
    tes_loss, tes_acc = model.evaluate(x, y, batch_size=batch_size)
    print("test loss is : ", tes_loss)
    print("test accuracy is : ", tes_acc)
    return tes_loss, tes_acc


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
    # 小数据集测试
    ori_data = ori_data[0:2000]

    # type_dict = {'本地生活': 0, '宠物生活': 1, '厨具锅具': 2, '电脑/办公': 3,
    #              '服饰鞋帽': 4, '家居家装': 5, '家用/商用家具': 6, '家用电器': 7,
    #              '家装建材': 8, '教育音像': 9, '母婴用品/玩具乐器': 10,
    #              '汽配用品': 11, '生鲜水果': 12, '食品/饮料/酒水': 13,
    #              '手机数码': 14, '图书杂志': 15, '箱包皮具': 16, '医药保健': 17,
    #              '音乐影视': 18, '运动户外': 19, '钟表礼品': 20, '珠宝饰品': 21}
    type_dict = {'本地生活': 0, '宠物生活': 1}

    # 打乱数据
    np.random.shuffle(ori_data)
    # 获得四个数据集
    tra_sams, tes_sams, tra_labs, tes_labs = divide_data(ori_data)
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
    tra_y = digitize_label(tra_labs, type_dict)
    tes_y = digitize_label(tes_labs, type_dict)

    print("------------------------------------数据预处理结束---------------------------------------")

    """ 拆散之后的代码
    """
    model = create_CNN_model(len(index_dict), len(type_dict), tok_dim=64)
    # model = create_DNN_model(len(index_dict), len(type_dict), tok_dim=64)
    # model = create_Text_CNN_model(len(index_dict), len(type_dict), tok_dim=64)
    # model = create_Fast_Text_model(len(index_dict), len(type_dict), tok_dim=64)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(tra_toks_index, tra_y, epochs=10, batch_size=64)
    tes_loss, tes_acc = model.evaluate(tra_toks_index, tra_y, batch_size=64)
    print("test loss is : ", tes_loss)
    print("test accuracy is : ", tes_acc)
    # 绘制模型训练结果图
    draw_result(history)


    """ 原代码
    
    # 选择模型
    model = choose_model("CNN", len(index_dict), len(type_dict))
    # 模型配置
    model_configure(model)
    # 模型训练
    history = train_model(model, tra_toks_index, tra_y, epochs=5, batch_size=1024)
    # 模型评估
    model_evaluate(model, tes_toks_index, tes_y, batch_size=1024)
    # 绘制模型训练结果图
    draw_result(history)
    """