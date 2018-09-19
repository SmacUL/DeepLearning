# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:39:25 2018

@author: SmacUL
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
    x_train = np.reshape(x_train, (len(x_train), fe_gap))
    x_test = np.reshape(x_test, (len(x_test), fe_gap))
    
#    x_train = np.reshape(x_train, (len(x_train), 1, fe_gap))
#    x_test = np.reshape(x_test, (len(x_test), 1, fe_gap))
    
    label = data[fe_gap + fo_gap - 1:]
    y_train = label[:-test_len]
    y_test = label[-test_len:]
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test



def create_model(input_shape):
    """ create empty LSTM model
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='mean_squared_error',
                  optimizer='adam')  
    return model
    


def train_model(model, x_train, y_train, epochs=100):
    """ 
    """
    model.fit(x_train, y_train, batch_size=64, epochs=epochs)



def forecast_evalute(model, x_test, y_test, data):
    """ 获得模型预测值并计算NMSE
    """
    y_test_pre = model.predict(x_test, batch_size=64)
    
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
    
    
#    NMSE=0
#    for i in range(y_test.shape[0]):
#        a = (y_test[i] - y_test_pre[i,0]) * (y_test[i] - y_test_pre[i,0])
#        b = (y_test[i] - y_test.mean()) *(y_test[i] -y_test.mean())
#        NMSE += a / b
#    
#    NMSE = (NMSE) / len(x_test) / len(x_test)    
        
        
        
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
import scipy.io as si
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

if __name__ == "__main__":
    URL = 'data00.mat'
    city_data_all = read_data(URL)
    
#    """
    test_len = 300
    fe_gap = 50
    fo_gap = 3
    epochs = 10
    for city_index in range(1):
        data = data_stabilize(city_data_all[city_index])
        x_train, y_train, x_test, y_test = create_trts_set(data,
                                                           test_len=test_len, 
                                                           fe_gap=fe_gap, 
                                                           fo_gap=fo_gap)
        model = create_model((fe_gap,))
        train_model(model, x_train, y_train, epochs=epochs)
        NMSE = forecast_evalute(model, x_test, y_test, city_data_all[city_index])
        print(NMSE)
#    """


    # 每次运行刷新前面的数据
    result_path = './res_DNN.txt'
    with open(result_path, 'w') as result:
        result.write("city_index fo_gap_val \t NMSE\r")
    
    for city_index in range(len(city_data_all)):
        for fo_gap_val in range(1, 22, 2):
            
            test_len = 300
            fe_gap = 50
            fo_gap = fo_gap_val     
            epochs = 10
            
            data = data_stabilize(city_data_all[city_index])
            x_train, y_train, x_test, y_test = create_trts_set(data,
                                                               test_len=test_len, 
                                                               fe_gap=fe_gap, 
                                                               fo_gap=fo_gap)
            model = create_model((fe_gap,))
            train_model(model, x_train, y_train, epochs=epochs)
            NMSE = forecast_evalute(model, x_test, y_test, city_data_all[city_index])     
            
            with open(result_path, 'a') as result:
                result.write("%d\t\t\t\t\t%d\t\t\t\t%.6f\r" % (city_index, fo_gap_val, NMSE))
                
            print("%d %d %.6f\r" % (city_index, fo_gap_val, NMSE))
            
        with open(result_path, 'a') as result:
            result.write("\rcity_index fo_gap_val \t NMSE\r")   





