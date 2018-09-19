# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 21:27:38 2018

@author: SmacUL
"""



def read_data():
    """ it will return a data list, size: (*, 5)
    
    抽取的5列数据：open close high low volume
    
    param:
        data: list, size: (*, 5)
    """
    source = ts.get_k_data\
        ('sh', ktype='D', autype='hfq',start='1993-01-01',end='2018-12-31')
    data_all = source.values
    data_ma = data_all[:, 1:6]
    
    data = []
    for i in range(data_ma.shape[0]):
        group = []
        for j in range(data_ma.shape[1]):
            group.append(data_ma[i, j])
        data.append(group)
    
    return data



def data_stabilize(data):
    """ 数据平稳处理
    
    数据以列为单位作归一化操作
    
    """ 
    min_max_list = []
    for i in range(len(data[0])):
        median = []
        for j in range(len(data)):
            median.append(data[j][i])
        
        min_num = min(median)
        max_num = max(median)
        min_max_list.append((min_num, max_num))
        
        for j in range(len(data)):
            a = (data[j][i] - min_num) / (max_num - min_num)
            data[j][i] = a
        
    return data, min_max_list



def create_trts_set(data, test_len, time_step=30, fo_gap=1):
    """ create train && test data set, x y both.
    
    这个方法中间会获得训练集标签 y_train 中最后一个收盘价，
    方便后面的分类计算
    
    """
    group_all = []
    for i in range(len(data) - time_step - fo_gap + 1):
        group = []
        for j in range(i, i + time_step):
            for k in range(len(data[0])):
                group.append(data[j][k])
        group_all.append(group)
    
    x_train = group_all[:-test_len]
    x_test = group_all[-test_len:]
    
    x_train = np.reshape(x_train, (len(x_train), time_step, len(data[0])))
    x_test = np.reshape(x_test, (len(x_test), time_step, len(data[0])))
    
    
    label = data[time_step + fo_gap - 1:]
    
    train_label = label[:-test_len]
    test_label = label[-test_len:]
    
    y_train = [train_label[i][1] for i in range(len(train_label))]
    y_test = [test_label[i][1] for i in range(len(test_label))]
    
    last_close = y_train[-1]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test, last_close



def create_model(input_shape, units=40, dropout=0):
    """ 生成空的LSTM模型
    """ 
    model = Sequential()
    model.add(LSTM(units=units, dropout=dropout, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')  
#    model.compile(loss='binary_crossentropy', optimizer='rmsprop')  
    return model


def train_model(model, x_train, y_train, epochs=100, batch_size=32):
    """ 训练模型
    """ 
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)



def get_forecast_value(model, x_test, batch_size=32):
    """ 获得模型预测值
    """
    y_test_pre = model.predict(x_test, batch_size=batch_size)
    return y_test_pre



def revert_data(y_data, min_num, max_num):
    """ 还原数据
    """
    for i in range(len(y_data)):
        y_data[i] = y_data[i] * (max_num - min_num) + min_num
    return y_data        

    

def classify_result_new(y_test_pre, y_test, last_close, class_gap=0.005):
    """ 分类器
    """
    y_test_pre_f = []
    for i in range(len(y_test_pre)):
        if i != 0:
            a = (y_test_pre[i] - y_test[i - 1]) / y_test[i - 1]
        else:
            a = (y_test_pre[i] - last_close) / last_close

        if a > class_gap:
            y_test_pre_f.append(1)
        elif a < -class_gap:
            y_test_pre_f.append(-1)
        else:
            y_test_pre_f.append(0)
    
    y_test_f = []
    for i in range(len(y_test)):
        if i != 0:
            a = (y_test[i] - y_test[i - 1]) / y_test[i - 1]
        else:
            a = (y_test[i] - last_close) / last_close
 
        if a > class_gap:
            y_test_f.append(1)
        elif a < -class_gap:
            y_test_f.append(-1)
        else:
            y_test_f.append(0)       
    
    return y_test_pre_f, y_test_f



def figure_CM(y_test_pre_f, y_test_f):
    """ 获得混淆矩阵的结果
    """
    cm = confusion_matrix(y_test_pre_f, y_test_f)
    print("回测天数", test_len)
    print('上涨召回率', cm[0, 0] / sum(cm[0, :]))
    print('平盘召回率', cm[1, 1] / sum(cm[1, :]))
    print('下跌召回率', cm[2, 2] / sum(cm[2, :]))
    print('上涨精度', cm[0, 0] / sum(cm[:, 0]))
    print('平盘精度', cm[1, 1] / sum(cm[:, 1]))
    print('下跌精度', cm[2, 2] / sum(cm[:, 2]))
    print('总体正确率率', (cm[0, 0] + cm[1, 1] + cm[2, 2]) / sum(sum(cm)))
    return cm



def figure_NMSE(y_test_pre, y_test):
    """ 计算NMSE
    """
#    NMSE = 0
#    for i in range(len(y_test)):
#        a = (y_test[i] - y_test_pre[i, 0]) * (y_test[i] - y_test_pre[i, 0])
#        b = (y_test[i] - y_test.mean()) * (y_test[i] - y_test.mean())
#        NMSE += a / b
#    
#    NMSE = (NMSE) / len(y_test) / len(y_test)  
    
    a_sum = 0
    b_sum = 0
    for i in range(len(y_test)):
        a = (y_test[i] - y_test_pre[i]) * (y_test[i] - y_test_pre[i])
        a_sum += a
        b = (y_test[i]) * (y_test[i])
        b_sum += b
        
    NMSE = a_sum / b_sum
    
    NMSE = 10 * m.log10(NMSE)
    
    print("model's NMSE is ", NMSE)
    return NMSE



import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts 
import scipy.io as si
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Reshape
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    
    hard_data = read_data()
    data, min_max_list = data_stabilize(hard_data)
    
    epochs = 100
    units = 25
    dropout = 0
    batch_size = 64
    
    test_len = 150
    time_step = 200
    fo_gap = 1
    class_gap = 0.005
    
    x_train, y_train, x_test, y_test, last_close = \
        create_trts_set(
                data, test_len=test_len, time_step=time_step, fo_gap=fo_gap)
    model = create_model((time_step, len(data[0])), 
                         units=units, dropout=dropout)
    train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)
    # 获得预测值
    y_test_pre = get_forecast_value(model, x_test, batch_size=batch_size)
    plt.plot(np.array(y_test_pre));plt.plot(np.array(y_test))
    # 计算NMSE
    NMSE = figure_NMSE(y_test_pre, y_test)
    # 还原归一化（预测值和真实值） 
    y_test_pre_rev = revert_data(y_test_pre, min_max_list[1][0], min_max_list[1][1])
    y_test_rev = revert_data(y_test, min_max_list[1][0], min_max_list[1][1])
#    plt.plot(np.array(y_test_pre_rev));plt.plot(np.array(y_test_rev))
    # 计算NMSE
    NMSE = figure_NMSE(y_test_pre_rev, y_test_rev)
    # 获得分类结果
    y_test_pre_cla, y_test_cla = \
        classify_result_new(
                y_test_pre_rev, y_test_rev, last_close, class_gap=class_gap)
    # 获得混淆矩阵
    CM = figure_CM(y_test_pre_cla, y_test_cla)
    
    
    
    