#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:54:31 2018

@author: smac-9
"""



def draw_pic(data):
    df = pd.DataFrame(data, columns=['value'])
    df.plot()



def read_data(URL):
    """ in this case, get data list of 12 city's power.
    
    return:
        city_data_all: a list contains 12 small list
    """
    data = si.loadmat(URL)['data']
    city_data_all = []
    for i in range(data.shape[0]):
        city_data = []
        for j in range(data.shape[1]):
            city_data.append(data[i][j])
        city_data_all.append(city_data)
    return city_data_all



def data_stabilize(data):
    """ 数据平稳处理
    
    param:
        data: it's a list, which size is 35043 * 1
    """
    min_num = min(data)
    max_num = max(data)
    
    new_data = []
    for i in range(len(data)):
        a = (data[i] - min_num) / (max_num - min_num)
        if a <= 0:
            a = abs(a) + 1
        new_data.append(a)
    return new_data



def create_trts_set(data, test_len, fe_gap=5, fo_gap=1):
    """ create train && test data set, x y both.
    
    param:
        data: it's a list, which size is 35043 * 1
    """
    group_all = []
    for i in range(len(data) - fe_gap - fo_gap + 1):
        group = data[i:i + fe_gap]
        group_all.append(group)
    x_train = group_all[:-test_len]
    x_test = group_all[-test_len:]
    
#    x_train = np.reshape(x_train, (len(x_train), fe_gap, 1))
#    x_test = np.reshape(x_test, (len(x_test), fe_gap, 1))
#    
    label = data[fe_gap + fo_gap - 1:]
    y_train = label[:-test_len]
    y_test = label[-test_len:]
#    y_train = np.array(y_train)
#    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test



def create_model():
    """ 
    """
    model = linear_model.LinearRegression()
    return model
    
    


def train_model(model, x_train, y_train):
    """ 
    """ 
    model.fit(x_train, y_train)
    res = model.coef_
    return res



def forecast_evalute(res, x_test, y_test, data):
    """ 获得预测值并计算NMSE
    """    
    y_test_pre = []
    for i in range(len(x_test)):
        y_f = 0
        for j in range(len(res)):
            y_f += res[j] * x_test[i][j]
        y_test_pre.append(y_f)
    
    plt.plot(np.array(y_test_pre));plt.plot(np.array(y_test))
    
    
    
    
    min_num = min(data)
    max_num = max(data)
    for i in range(len(y_test)):
        if y_test[i] == 1:
            y_test[i] = 0
        y_test[i] = y_test[i] * (max_num - min_num) + min_num
        if y_test_pre[i] == 1:
            y_test_pre[i] = 0
        y_test_pre[i] = y_test_pre[i] * (max_num - min_num) + min_num
    
    # calculate NMSE
    a_sum = 0
    b_sum = 0
    for i in range(len(y_test)):
        a = (y_test[i] - y_test_pre[i]) * (y_test[i] - y_test_pre[i])
        a_sum += a
        b = (y_test[i]) * (y_test[i])
        b_sum += b
        
    NMSE = a_sum / b_sum
    
    NMSE = 10 * m.log10(NMSE)
    
    return NMSE



import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts 
import scipy.io as si
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    URL = 'data00.mat'
    city_data_all = read_data(URL)

    """ 单次处理    
    test_len = 300
    fe_gap = 50
    fo_gap = 5
    
    for city_index in range(1):
        data = data_stabilize(city_data_all[city_index])
        x_train, y_train, x_test, y_test = create_trts_set(data,
                                                           test_len=test_len, 
                                                           fe_gap=fe_gap, 
                                                           fo_gap=fo_gap)
        model = create_model()
        res = train_model(model, x_train, y_train)
        NMSE = forecast_evalute(res, x_test, y_test, city_data_all[city_index])
    
        print(NMSE)    
    """
        
    # 每次运行刷新前面的数据
    result_path = './res_ARX.txt'
    with open(result_path, 'w') as result:
        result.write("city_index fo_gap_val \t NMSE\r")
    
    for city_index in range(len(city_data_all)):
        for fo_gap_val in range(1, 22, 2):
            test_len = 300
            fe_gap = 50
            fo_gap = fo_gap_val          
            
            data = data_stabilize(city_data_all[city_index])
            x_train, y_train, x_test, y_test = create_trts_set(data,
                                                               test_len=test_len, 
                                                               fe_gap=fe_gap, 
                                                               fo_gap=fo_gap)
            model = create_model()
            res = train_model(model, x_train, y_train)
            NMSE = forecast_evalute(res, x_test, y_test, city_data_all[city_index])
    
            with open(result_path, 'a') as result:
                result.write("%d\t\t\t\t\t%d\t\t\t\t%.6f\r" % (city_index, fo_gap_val, NMSE))
                
            print("%d %d %.6f\r" % (city_index, fo_gap_val, NMSE))
            
        with open(result_path, 'a') as result:
            result.write("\rcity_index fo_gap_val \t NMSE\r")        

    
