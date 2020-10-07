"""
Created on Monday, Oct. 8, 2019

@author: whyang
"""
# -*- coding: utf-8 -*-

###
# declare needed packages
#
import pandas as pd
from flask import Flask
from flask_restplus import Api, Resource, fields, marshal
from server.restAPI import server # REST UI's server for popvar (population variation)
from models.popvar_model import * # REST UI's data model(schema) for popvar (population variation)

###
# initiate REST UI's server and API modules
#
app, api = server.app, server.api

###
# API structure of the enquiry for indigenous peoples population and its variation w.r.t. CIP's datasets
# URL of API: 
#     cip_pop_var 1.0 ----- tribe ----- 100_102  # query of population w.r.t tribe
#                                 ----- 100_103
#                                 ----- 100_107
#                                 ----- 102_103
#                                 ----- 102_107
#                                 ----- 103_107
#     cip_pop_var 1.0 ----- area  ----- 100_102 # query of population w.r.t. each city and county
#                                 ----- 100_103
#                                 ----- 100_107
#                                 ----- 102_103
#                                 ----- 102_107
#                                 ----- 103_107
# URL of API on Swagger UI: cip_pop_var/api
# base URL of API: cip_pop_var/1.0/
# adopt namespace in the api of flask to practice the above cip_pop_var API structure 
#
_ns_tribe = api.namespace('tribe', description='原住民族人口變化(100-107): 族')
_ns_area = api.namespace('area', description='原住民族人口變化(100-107): 縣市')


#######################################################################################
# declare functions
#######################################################################################
###
# remove leading and trailing characters of each value across all cells in dataframe
#
def trim_all_cells(df):
    # trim whitespace from ends of each value across all series in dataframe
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)

###
# make sure the queried tribe is existed
#
def abort_if_tribe_doesnt_exist(tribe):
    if tribe not in TRIBES:
        api.abort(404, "找不到該原住民族的資料".format(tribe))
###
# make sure the queried area (city or county) is existed
#
def abort_if_area_doesnt_exist(area):
    if area not in AREAS:
        api.abort(404, "找不到該縣市的資料".format(tribe))       
#######################################################################################
# url routing for the /tribe and /area 
#######################################################################################

#parser = api.parser()
#parser.add_argument('tribe', type=str, required=True, help='原住民族族名(例如,阿美族)', location='form')

###
# the period of the years 102 to 103
#        
###
# /tribe/102_103 query all tribes' variation info. of the indigenous population 
# 所有原住民族在各縣市之人口數變化'
#
@_ns_tribe.route('/102_103/')
class TribeList_102_103(Resource):
    @_ns_tribe.marshal_with(var_tribe_list_model) # declare corresponded data model(schema) for response 
    @_ns_tribe.doc(description='目前原住民族人口統計資料集內容，可以查詢到的所有原住民族有: {0}'.format(', '.join(TRIBES.keys())))
    @_ns_tribe.doc(
            responses={
                    200: 'Success',
                    400: 'Validation Error'
                    }
            )
    def get(self):
        '''查詢各縣市(區域)內，每個原住民族人口的年度變動數量 (102年~103年)'''            
        ###
        # read the data between the year of 102 and 103
        #
        with open('..\\data\\population-var-102-103.csv', 'r', encoding='utf-8', newline='') as csvfile:
            df_102_103 = pd.read_csv(
                csvfile,
                header = 0,
                usecols = ['區域別',
                #usecols = ['日期區間', '區域別', '總計',
                           '阿美族', '泰雅族', '排灣族', '布農族', '魯凱族', '卑南族', '鄒族', '賽夏族',
                           '雅美族', '邵族', '噶瑪蘭族', '太魯閣族', '撒奇萊雅族', '賽德克族',
                           '拉阿魯哇族', '卡那卡那富族', '尚未申報'],
                index_col = '區域別', # indexing as the names of city or county in AREAS
                verbose = True,
                skip_blank_lines = True)
            df_102_103 = trim_all_cells(df_102_103) # trim whitespace from each cell in dataframe
            
            ###
            # transpose dataframe to coincide to the data model of response
            #
            df = df_102_103.T # transpose()
            
            ###
            # add on tribe names w.r.t. '族別' column
            #
            df['族別'] = ['阿美族', '泰雅族', '排灣族', '布農族', '魯凱族', '卑南族', '鄒族', 
              '賽夏族', '雅美族', '邵族', '噶瑪蘭族', '太魯閣族', '撒奇萊雅族', 
              '賽德克族','拉阿魯哇族', '卡那卡那富族', '尚未申報']
            
            ###
            # arrange responded values w.r.t. the data model of response
            # row_1 (column_1_name: cell_value,..., column_n_name: cell_value),...,row_n (...)
            # area = city/county name, tribe_name = value, ...
            #
            df1 = df.to_dict('records') # arrange responded values w.r.t. the data model of response
            
            ###
            # send out response
            #
            return df1  

###
# /tribe/100_103/<string:tribe> query one specific tribe's variation info. of the indigenous population 
#
@_ns_tribe.route('/102_103/<string:tribe>')
@_ns_tribe.doc(responses={404: '找不到該原住民族的資料'}, params={'tribe': '原住民族族名'})
class Tribe_102_103(Resource):
    @_ns_tribe.marshal_with(var_tribe_model) # declare corresponded data model(schema) for response 
    @_ns_tribe.doc(description='目前原住民族人口統計資料集內容，可以提供查詢的原住民族有: {0}'.format(', '.join(TRIBES.keys())))
    @_ns_tribe.doc(
            responses={
                    200: 'Success',
                    400: 'Validation Error'
                    }
            )
    def post(self, tribe): #get(self, tribe):
        '''查詢某一個特定原住民族(如阿美族)，在各縣市(區域)人口的年度變動數量 (102年~103年)'''
        ###
        # inspect the correctness of the queried tribe's name
        #
        abort_if_tribe_doesnt_exist(tribe)
        
        ###
        # read the data between the year of 102 and 103
        #       
        with open('..\\data\\population-var-102-103.csv', 'r', encoding='utf-8', newline='') as csvfile:
            df_102_103 = pd.read_csv(
                csvfile,
                header = 0,
                usecols = ['區域別', tribe], # only one column for the queried tribe
                #usecols = ['日期區間', '區域別', '總計',
                           #'阿美族', '泰雅族', '排灣族', '布農族', '魯凱族', '卑南族', '鄒族', '賽夏族',
                           #'雅美族', '邵族', '噶瑪蘭族', '太魯閣族', '撒奇萊雅族', '賽德克族',
                           #'拉阿魯哇族', '卡那卡那富族', '尚未申報'],
                verbose = True,
                skip_blank_lines = True)
            df_102_103 = trim_all_cells(df_102_103) # trim whitespace from each cell in dataframe
        
        ###
        # map the dataframe's columns of '區域別' and tribe_name into the names w.r.t. the used data model(schema)
        #
        df_102_103.rename({'區域別': 'area', tribe: 'amount'}, axis='columns', inplace=True) # axis=1

        ###
        # arrange responded values w.r.t. the data model of response
        # row_1 (area: cell_value('區域別'), amount: cell_value(tribe)),...,row_n()
        # area = city/county name, amount = pop_var of the tribe in the city/county
        #        
        df = df_102_103.to_dict('records')
 
        ###
        # send out response
        #
        return df

###
# /area/102_103 query the tribes' variation info. of the indigenous population in the area (city/county) 
# 各縣市(區域)內之原住民族人口數變化'
#
@_ns_area.route('/102_103/')
class AreaList_102_103(Resource):
    @_ns_area.marshal_with(var_area_list_model) # declare corresponded data model(schema) for response 
    @_ns_area.doc(description='目前原住民族人口統計資料集內容，可以查詢的縣市有: {0}'.format(', '.join(AREAS.keys())))
    @_ns_area.doc(
            responses={
                    200: 'Success',
                    400: 'Validation Error'
                    }
            )
    def get(self):
        '''查詢原住民族人口統計數據，在各縣市的年度變動數量 (102年~103年)'''            
        ###
        # read the data between the year of 102 and 103
        #
        with open('..\\data\\population-var-102-103.csv', 'r', encoding='utf-8', newline='') as csvfile:
            df_102_103 = pd.read_csv(
                csvfile,
                header = 0,
                usecols = ['區域別','總計',
                #usecols = ['日期區間', '區域別', '總計',
                           '阿美族', '泰雅族', '排灣族', '布農族', '魯凱族', '卑南族', '鄒族', '賽夏族',
                           '雅美族', '邵族', '噶瑪蘭族', '太魯閣族', '撒奇萊雅族', '賽德克族',
                           '拉阿魯哇族', '卡那卡那富族', '尚未申報'],
                #index_col = '區域別', # skip to index with any column
                verbose = True,
                skip_blank_lines = True)
            df_102_103 = trim_all_cells(df_102_103) # trim whitespace from each cell in dataframe

            ###
            # arrange responded values w.r.t. the data model of response
            # row_1 (column_1_name: cell_value,..., column_n_name: cell_value),...,row_n (...)
            # 區域別 = city/county name, 總計 = amount, tribe_name = value, ...
            #
            df1 = df_102_103.to_dict('records')
            
            ###
            # send out response
            #
            return df1

###
# /area/102_103/<string:area> query the tribes' variation info. of the indigenous population in the specific area (city/county) 
#
@_ns_area.route('/102_103/<string:area>')
@_ns_area.doc(responses={404: '找不到該縣市的資料'}, params={'area': '縣市名稱'})
class Area_102_103(Resource):
    @_ns_area.marshal_with(var_area_model) # declare corresponded data model(schema) for response
    @_ns_area.doc(description='目前原住民族人口統計資料集內容，可以提供查詢的區域(縣市)有: {0}'.format(', '.join(AREAS.keys())))
    @_ns_area.doc(
            responses={
                    200: 'Success',
                    400: 'Validation Error'
                    }
            )
    def post(self, area): #get(self, area):
        '''查詢某一個縣市(如新北市)內，各原住民族人口的年度變動數量 (102年~103年)'''
        ###
        # inspect the correctness of the queried area's name
        #
        abort_if_area_doesnt_exist(area)
        
        ###
        # read the data between the year of 102 and 103
        #     
        with open('..\\data\\population-var-102-103.csv', 'r', encoding='utf-8', newline='') as csvfile:
            df_102_103 = pd.read_csv(
                csvfile,
                header = 0,
                usecols = ['區域別',
                #usecols = ['日期區間', '區域別', '總計',
                           '阿美族', '泰雅族', '排灣族', '布農族', '魯凱族', '卑南族', '鄒族', '賽夏族',
                           '雅美族', '邵族', '噶瑪蘭族', '太魯閣族', '撒奇萊雅族', '賽德克族',
                           '拉阿魯哇族', '卡那卡那富族', '尚未申報'],
                #index_col = '區域別', # skip to index with any column
                verbose = True,
                skip_blank_lines = True)
            df_102_103 = trim_all_cells(df_102_103) # trim whitespace from each cell in dataframe

            ###
            # to deal with two counties' names (桃園縣 and 桃園市)
            # in the year of 100~102 and 102~103, 桃園縣 is collected in the dataset (while 桃園市 is not included because it is 省轄市)
            # however, 桃園市(直轄市) contains the number of the both of 桃園縣 and 桃園市(省轄市) in the year of 103~107
            #
            if (area == '桃園市'):
                area = '桃園縣'
            # unify to use '臺' instead of '台'   
            if (area == '台北市'): 
                area = '臺北市'
            elif (area == '台中市'): 
                area = '臺中市'   
            elif (area == '台南市'): 
                area = '臺南市'               
            elif (area == '台東縣'): 
                area = '臺東縣' 

            ###
            # to screen out the number w.r.t. each tribe of the specific city/country
            #             
            filter = df_102_103['區域別'] == area # filter context in terms of the name of the city/county
            # construct the columns of the output response structure
            y_102_103_col= ['阿美族', '泰雅族', '排灣族', '布農族', '魯凱族', '卑南族', '鄒族', '賽夏族',
                            '雅美族', '邵族', '噶瑪蘭族', '太魯閣族', '撒奇萊雅族', '賽德克族',
                            '拉阿魯哇族', '卡那卡那富族', '尚未申報']
            # construct the row of the output response structure
            y_102_103 = [df_102_103.loc[filter, '阿美族'].values[0],
                         df_102_103.loc[filter, '泰雅族'].values[0],
                         df_102_103.loc[filter, '排灣族'].values[0],
                         df_102_103.loc[filter, '布農族'].values[0],
                         df_102_103.loc[filter, '魯凱族'].values[0],
                         df_102_103.loc[filter, '卑南族'].values[0],
                         df_102_103.loc[filter, '鄒族'].values[0],
                         df_102_103.loc[filter, '賽夏族'].values[0],
                         df_102_103.loc[filter, '雅美族'].values[0],
                         df_102_103.loc[filter, '邵族'].values[0],
                         df_102_103.loc[filter, '噶瑪蘭族'].values[0],
                         df_102_103.loc[filter, '太魯閣族'].values[0],
                         df_102_103.loc[filter, '撒奇萊雅族'].values[0],
                         df_102_103.loc[filter, '賽德克族'].values[0],
                         df_102_103.loc[filter, '拉阿魯哇族'].values[0],
                         df_102_103.loc[filter, '卡那卡那富族'].values[0],
                         df_102_103.loc[filter, '尚未申報'].values[0]]
            ###
            # map the dataframe's columns of 'tribe' and 'amount' into the above response structure of column and row w.r.t. the data model(schema)
            # 
            dict = {"tribe": y_102_103_col, "amount": y_102_103}
            df = pd.DataFrame(dict)

            ###
            # arrange responded values w.r.t. the data model of response
            # row_1 (tribe: cell_value, amount: cell_value(tribe)),...,row_n()
            # tribe = tribe's name, amount = pop_var of the tribe in the city/county
            #                          
            df1 =df.to_dict('records')
            
            ###
            # send out response
            #
            return df1     
