#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 09:14:34 2018

Bug Fixed on 2018年 07月 30日 星期一 19:11:01 CST by Smac-9
:
    method create_trts_set unable to accept different values of 
    parameter fe_gap except 5.
    
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


#""" do nothing """
    
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
    

    return x_train, y_train, x_test, y_test



def create_model():
    """ 
    """
    
    reg = linear_model.LinearRegression()
    return reg
    
    


def train_model(model, x_train, y_train):
    """ 
    """ 
    model.fit(x_train, y_train)
    res = model.coef_
    return res


def forecast_evalute(res, x_test, y_test):
    """ 获得预测值并计算NMSE
    """    
    y_test_pre = []
    for i in range(len(x_test)):
        y_f = 0
        for j in range(len(res)):
            y_f += res[j] * x_test[i][j]
        y_test_pre.append(y_f)
    
    
#    print(y_test_pre)
    
    # calculate mean
    y_test_mean = sum(y_test) / len(y_test)
    
    # calculate NMSE
    NMSE = 0
    for i in range(len(y_test)):
        a = (y_test[i] - y_test_pre[i]) * (y_test[i] - y_test_pre[i])
        b = (y_test[i] - y_test_mean) * (y_test[i] - y_test_mean)
        NMSE += a / b
    
    
    NMSE = (NMSE) / len(x_test) / len(x_test)    
#    print(NMSE)
    return NMSE





import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts 
from sklearn import linear_model

if __name__ == "__main__":
    data = read_data()
    data = data_stabilize(data)
    
    
    NMSE_list = []
    for i, value in enumerate(range(1, 10, 1)):
        test_len = 100
        fe_gap = value
        fo_gap = 1
        # every kind of suituation will be run three times
        # to avoid occasionality,
        # then it will calculate the mean value of these three datas
        for j in range(3):
            x_train, y_train, x_test, y_test = create_trts_set(
                                                    data,
                                                    test_len=test_len, 
                                                    fe_gap=fe_gap, 
                                                    fo_gap=fo_gap
                                                )
            model = create_model()
            res = train_model(model, x_train, y_train)
            NMSE = forecast_evalute(res, x_test, y_test)
            NMSE_list.append(NMSE)
            
        
        sum_num = 0
        for j in range(i * 4, i * 4 + 3):
            sum_num += NMSE_list[j]
        NMSE_list.append(sum_num / 3)
    
    for i in range(len(NMSE_list)):
        print(NMSE_list[i])
    
    
   