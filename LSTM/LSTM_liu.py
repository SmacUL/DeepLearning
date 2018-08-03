#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 09:14:13 2018

Bug Fixed on 2018年 07月 30日 星期一 19:11:01 CST by smac-9
:
    Method create_trts_set unable to accept different values of 
    parameter fe_gap except 5.
    
Program Modified on 2018年 07月 31日 星期二 10:42:52 CST by smac-9
:
    Modified the method create_model, 
    using binary_crossentropy loss function to replace mean_squared_error,
    and using rmsprop optimizer to replace adam.
    In this way, 
    the LSTM model is able to get more smooth and better result.

Program Modified on 2018年 08月 01日 星期三 14:50:14 CST by smac-9
:
    Add method forecast_devide.

Program Modified on 2018年 08月 03日 星期五 09:06:59 CST by smac-9
:
    Add method create_model_LSTMCNN.
    The model created by this method
    will contain a basic LSTM mod and a basic CNN mod.

@author: smac-9
"""



def draw_pic(data):
    df = pd.DataFrame(data, columns=['value'])
    df.plot()



def read_data():
    """ it will return a data list, size: (*, 5)
    
    param:
        data: list, size: (*, 5)
    """
    source = ts.get_k_data\
        ('sh', ktype='D', autype='hfq',start='1993-01-01',end='2018-12-31')
    data_all = source.as_matrix()
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
    """ 
#""" method A """
    for i in range(len(data[0])):
        median = []
        for j in range(len(data)):
            median.append(data[j][i])
        
        min_num = min(median)
        max_num = max(median)
        
        # data * (max_num - min_num) + min_num 
        
#        print(min_num, max_num)
        
        for j in range(len(data)):
            a = (data[j][i] - min_num) / (max_num - min_num)
            if a <= 0:
                a = abs(a) + 1
            data[j][i] = a
    return data



#""" method B """

#    new_data = []
#    
#    for i in range(len(data) - 1):
#        group = []
#        for j in range(len(data[0])):
#            group.append(m.log(data[i][j]) - m.log(data[i + 1][j]))
#        new_data.append(group)
#    return new_data
        



#""" method C """
    
#    new_data = []
#    for i in range(len(data)):
#        group = []
#        for j in range(len(data[0])):
#            group.append(m.log(data[i][j]))
#        new_data.append(group)
#    return new_data



#"""" do nothing """

#    return data



def create_trts_set(data, test_len, fe_gap=5, fo_gap=1):
    """ create train && test data set, x y both.
    """
    group_all = []
    for i in range(len(data) - fe_gap - fo_gap + 1):
        group = []
        for j in range(i, i + fe_gap):
            for k in range(len(data[0])):
                group.append(data[j][k])
        group_all.append(group)
    
    
    
    x_train = group_all[:-test_len]
    x_test = group_all[-test_len:]
    x_train = np.reshape(x_train, (len(x_train), 1, len(x_train[0])))
    x_test = np.reshape(x_test, (len(x_test), 1, len(x_test[0])))
    
    
    label = data[fe_gap + fo_gap - 1:]
    train_label = label[:-test_len]
    test_label = label[-test_len:]
    
    y_train = []
    for i in range(len(train_label)):
#        y_train.append(train_label[i][1] - train_label[i][0])
        y_train.append(train_label[i][1])
    
    y_test = []
    for i in range(len(test_label)):
#        y_test.append(test_label[i][1] - test_label[i][0])
        y_test.append(test_label[i][1])
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    return x_train, y_train, x_test, y_test



def create_model(input_shape):
    """ create empty LSTM model
    """ 
    model = Sequential()
    model.add(LSTM(units=40, dropout=0, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
#    model.compile(loss='mean_squared_error', optimizer='adam')  
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')  
    return model
    


def create_model_LSTMCNN(input_shape):
    model = Sequential()
    model.add(LSTM(units=40, dropout=0, input_shape=input_shape))
    
    model.add(Reshape((40, 1)))
    model.add(layers.Conv1D(16, (20), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(16, (20), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')  
    return model



def train_model(model, x_train, y_train, epochs=100):
    """ 
    """ 
    model.fit(x_train, y_train, batch_size=32, epochs=epochs)



def forecast_evalute(model, x_test, y_test):
    """ 获得模型预测值并计算NMSE
    """
    y_test_pre = model.predict(x_test, batch_size=32)
    
    print(y_test_pre)
    print(y_test)
    
    plt.plot(np.array(y_test_pre));plt.plot(np.array(y_test))
    
    
    NMSE = 0
    for i in range(y_test.shape[0]):
        a = (y_test[i] - y_test_pre[i, 0]) * (y_test[i] - y_test_pre[i, 0])
        b = (y_test[i] - y_test.mean()) * (y_test[i] - y_test.mean())
        NMSE += a / b
    
    
    NMSE = (NMSE) / len(x_test) / len(x_test)    
    return NMSE



def forecast_devide(model, x_test, y_test):
    """
    """
    y_test_pre = model.predict(x_test, batch_size=32)
    plt.plot(np.array(y_test_pre));plt.plot(np.array(y_test))

    class_gap = 0.005
    spe_num = sum(y_test) / len(y_test)
    y_test_pre_f = []
    for i in range(len(y_test_pre)):
        if y_test_pre[i] > (spe_num + class_gap):
            y_test_pre_f.append(1)
        elif y_test_pre[i] < (spe_num - class_gap):
            y_test_pre_f.append(-1)
        else:
            y_test_pre_f.append(0)
    
    y_test_f = []
    for i in range(len(y_test)):
        if y_test[i] > (spe_num + class_gap):
            y_test_f.append(1)
        elif y_test[i] < (spe_num - class_gap):
            y_test_f.append(-1)
        else:
            y_test_f.append(0)
    
    cm = confusion_matrix(y_test_pre_f, y_test_f)
    return cm



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
    data = read_data()
    data = data_stabilize(data)
    epochs = 100

#    test_len = 300
#    fe_gap = 1
#    fo_gap = 1
#    
#    x_train, y_train, x_test, y_test = \
#        create_trts_set(
#            data, test_len=test_len, fe_gap=fe_gap, fo_gap=fo_gap
#        )
#    model = create_model((1, len(data[0]) * fe_gap))
#    train_model(model, x_train, y_train, epochs=epochs)
#    
#    NMSE = forecast_evalute(model, x_test, y_test)
#    print(NMSE)

#    cm = forecast_devide(model, x_test, y_test)
#    print(cm)
#    print("回测天数", test_len)
#    print('上涨召回率',cm[0,0]/sum(cm[0,:]))
#    print('平盘召回率',cm[1,1]/sum(cm[1,:]))
#    print('下跌召回率',cm[2,2]/sum(cm[2,:]))
#    print('上涨精度',cm[0,0]/sum(cm[:,0]))
#    print('平盘精度',cm[1,1]/sum(cm[:,1]))
#    print('下跌精度',cm[2,2]/sum(cm[:,2]))
#    print('总体正确率率',(cm[0,0]+cm[1,1]+cm[2,2])/sum(sum(cm)))
    
   


    NMSE_list = []
    for i, value in enumerate(range(1, 10, 1)):
        test_len = 100
        fe_gap = value
        fo_gap = 1
        # every kind of suituation will be run three times
        # to avoid occasionality,
        # then it will calculate the mean value of these three datas.
        for j in range(3):
            x_train, y_train, x_test, y_test = \
                create_trts_set(
                    data, test_len=test_len, fe_gap=fe_gap, fo_gap=fo_gap
                )
            model = create_model((1, 5 * fe_gap))
            train_model(model, x_train, y_train, epochs=epochs)
            NMSE = forecast_evalute(model, x_test, y_test)
            NMSE_list.append(NMSE)
            
        sum_num = 0
        for j in range(i * 4, i * 4 + 3):
            sum_num += NMSE_list[j]
        NMSE_list.append(sum_num / 3)
    
    for i in range(len(NMSE_list)):
        print(NMSE_list[i])
    
 
    