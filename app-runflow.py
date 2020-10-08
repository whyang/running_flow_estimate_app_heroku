# -*- coding: utf-8 -*-
"""
Created on Oct. 7, 2020
使用馬拉松賽事跑者人流分析與預測模型，進行預測 (deploy Flask on a cloud web platform, i.e., Heroku)
Raw data: 田中馬拉松(2017年、2018年、2019年)，全馬組跑者成績紀錄，建立之預測模型
Input: 感應點距離(離起點)、馬拉松舒適指數(環境因子、體感因子等9項)、
        觀測時間(起跑後通過某個感應點時間(秒), delta_time, i.e., 預測時間(成績))、
        全馬參賽(或報名)人數、鳴槍時間(秒)
Output: 預測人數(某個感應點，目前鳴槍時間(秒)後
@author: Wen-Hsin Yang
"""

import os
import time
import numpy as np
import pandas as pd
import joblib #pkl模型 format
from flask import Flask, render_template, request
from datetime import datetime

def revRunningPerformanceToTime(delta=0, base=0):
    timeStamp = delta + base
    #timeStamp //= 1e3
    timearr = time.gmtime(timeStamp) #由於沒有指定timezone，原先之時間戳轉換為utc時間的struct_time會主動調整為(utc+8)，使用gmtime會調整本地時間差8個小時
    otherStyleTime = time.strftime("%H:%M:%S", timearr)
    #print('*** 跑步成績 (現在觀測時間) = ', timeStamp, '(sec.)', otherStyleTime)
    return otherStyleTime

# get the path in terms of the base directory
base_dir = os.path.dirname(__file__)
app = Flask(__name__, 
            static_folder=os.path.join(base_dir, 'static'),
            template_folder=os.path.join(base_dir, 'templates'))

@app.route('/')
def home():
	return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():   
    ###
    # step 1: load the trained model
    ##
    # load existing model of the random forest regression: currently is based on the Tianzhong Marathon 2017~2019
    print('** load the existing model  (joblib)')
    model_dir = os.path.join(base_dir, 'model') # the path for storing models
    model = os.path.join(model_dir, 'rf_reg_flow(joblib).pkl') 
    print('model = {a}'.format(a=model))
    rf_reg = joblib.load(model)
   
    # load existing model of the XGBoost regression: currently is based on the Tianzhong Marathon 2017~2019
    model = os.path.join(model_dir, 'xgboost_reg_flow(joblib).pkl') 
    print('model = {a}'.format(a=model))
    xgboost_reg = joblib.load(model)
    
    ###
    # step 2: read configuration context
    ##
    # read into configuration parameter (i.e., the range of each features 
    config = os.path.join(model_dir, 'config.xlsx') # the configuration file
    df = pd.read_excel(config,
                       usecols=['鳴槍時間', '速度', '距離', '溫度', '濕度', '熱中暑危險係數',
                                '空氣品質指標', '細懸浮微粒', '蒲福風級', '小時雨量',
                                '預測時間', '預測人數(最多)', '全馬參賽(或報名)人數'])     
    # display values in the configuration file     
    range_commencetime = df.loc[0, '鳴槍時間']
    print('commence_time = ', range_commencetime)
    range_speed = df.loc[0, '速度'] 
    print('range_speed = ', range_speed)
    range_dist = df.loc[0, '距離']
    print('range_dist = ', range_dist)
    range_runscore = df.loc[0, '預測時間']
    print('range_runscore = ', range_runscore)
    range_temp = df.loc[0, '溫度']
    print('range_temp = ', range_temp)
    range_hum = df.loc[0, '濕度']
    print('range_hum = ', range_hum)
    range_hi = df.loc[0, '熱中暑危險係數']
    print('range_hi = ', range_hi)
    range_aqi = df.loc[0, '空氣品質指標']
    print('range_aqi = ', range_aqi)
    range_pm = df.loc[0, '細懸浮微粒']
    print('range_pm = ', range_pm)
    range_wr = df.loc[0, '蒲福風級']
    print('range_wr = ', range_wr)
    range_rain = df.loc[0, '小時雨量']
    print('range_rain = ', range_rain)
    range_runflow = df.loc[0, '預測人數(最多)'] # maximal number of runners in the training data 
    print('range_runflow = ', range_runflow)
    num_runner = df.loc[0, '全馬參賽(或報名)人數'] # 預測該場全馬賽事之參賽(或報名)人數
    print('num_runner = ', num_runner)
       
    ###
    # step 3: get the input arguments which are declared for conducting prediction
    ##
    # read thru the RESTful API of POST
    if request.method == 'POST':
        runner = request.values['runner']
        #
        commence_time_ = request.values['commence_time']
        commence_time = datetime.strptime(commence_time_, '%H:%M:%S')
        #
        observed_time_ = request.values['observed_time']
        observed_time = datetime.strptime(observed_time_, '%H:%M:%S')
        time_ = observed_time - commence_time # calculate the duration time from commence
        score = time_.seconds # transform to the unit of second
        #
        distance = request.values['distance']
        temp = request.values['temp']
        hum = request.values['hum']
        heatindex = request.values['heatindex']
        AQI = request.values['AQI']
        PM = request.values['PM']
        WR = request.values['WR']
        rain = request.values['rain']
                
        # normalize each value of the input argument, i.e., each feature's value
        distance_ = float(distance) / range_dist
        score_ = float(score) / range_runscore
        speed_ = ((float(distance) / 1000) / (float(score) / 3600)) / range_speed
        temp_ = float(temp) / range_temp
        hum_ = float(hum) / range_hum
        heatindex_ = float(heatindex) / range_hi
        AQI_ = float(AQI) / range_aqi
        PM_ = float(PM) / range_pm
        WR_ = float(WR) / range_wr
        rain_ = float(rain) / range_rain
        runner = float(runner)
        num_runner = abs(int(runner))
        # transform the commence time to the unit of second from the time 00:00:00
        based_time = datetime.strptime('00:00:00', '%H:%M:%S') 
        range_commencetime = (commence_time - based_time).seconds
        
        # construct arguments used by prediction model
        pred = np.array([[speed_, distance_, temp_, hum_, heatindex_, AQI_, PM_, WR_, rain_, score_]])                          
        x_pred = pd.DataFrame(data=pred, 
                              columns=['速度', '距離', '溫度', '濕度', '熱中暑危險係數',
                                       '空氣品質指標', '細懸浮微粒', '蒲福風級', '小時雨量', '預測時間'])
        print('x_pred = ', x_pred) 
        
        ###
        # step 4: conduct regression with both of Random Forest and XGBoost algorithms
        ##               
        # get the prediction results (random forest regression) respecting to the input x_pred
        y_pred = rf_reg.predict(x_pred)
        print('model(random forest regression)')
        print('runner_flow = {a} '.format(a=y_pred))
        print('runner_flow = {a} '.format(a=y_pred * num_runner))
        
        # get the prediction results (XGBoost regression) respecting to the input x_pred    
        y_pred_1 = xgboost_reg.predict(x_pred)
        print('model(xgboost regression)')
        print('runner_flow = {a} '.format(a=y_pred_1))
        print('runner_flow = {a} '.format(a=y_pred_1 * num_runner))
                
        # we adopt the predicted result of the XGBoost regression
        print('print out (XGBoosting regression')
        # forecasted number of the running flow
        output_flow = abs(int(y_pred_1 * num_runner))
        
        '''
        # transform the observed time(second) to the timestamp of the hh:mm:ss respecting to that day of the marathon event
        score_time = score_ * range_runscore
        output_score = revRunningPerformanceToTime(delta=score_time, base=range_commencetime)
        print('output flow={a}, time={b} {c}'.format(a=output_flow, b=int(score_time), c=output_score))
        output_commence_time = revRunningPerformanceToTime(delta=range_commencetime)
        print('鳴槍時間={a}'.format(a=output_commence_time))     
        '''
        
    # send out the prediction results and their corresponding inputs of features
    return render_template('output.html', 
                           prediction_flow=output_flow,
                           #
                           distance=distance,
                           observed_time=observed_time_,
                           runner=runner,
                           commence_time=commence_time_,
                           duration=score,
                           #
                           temp=temp,  
                           hum=hum, 
                           heatindex=heatindex, 
                           AQI=AQI,
                           PM=PM,
                           WR=WR, 
                           rain=rain
                           )
# end of def predict():
    
if __name__ == '__main__':
    # initiate web framework in terms of the flask
    #pp.run(debug=False, host='https://running-flow-estimate.herokuapp.com') # for deploying on Heroku 
    app.run(debug=True, host='0.0.0.0', port=5000)
    #app.run(debug=False, host='0.0.0.0', port=5000) 
###
# end of file
##