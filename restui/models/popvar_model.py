"""
Created on Monday, Sep. 17, 2019

@author: whyang
"""
# -*- coding: utf-8 -*-
from flask_restplus import fields
from server.restAPI import server

app, api = server.app, server.api

TRIBES = {
        '阿美族': 'Amis',
        '泰雅族': 'Atayal',
        '排灣族': 'Paiwan',
        '布農族': 'Bunun',
        '魯凱族': 'Rukai',
        '卑南族': 'Pinuyumayan',
        '鄒族': 'Tsou',
        '賽夏族': 'Saisiyat',
        '雅美族': 'Yami',
        '邵族': 'Thao',
        '噶瑪蘭族': 'Kavalan',
        '太魯閣族': 'Truku',
        '撒奇萊雅族': 'Sakizaya',
        '賽德克族': 'Seediq',
        '拉阿魯哇族': 'Hla’alua',
        '卡那卡那富族': 'Kanakanavu',
        '尚未申報': 'other'
        }
                           
AREAS = {
        '新北市': 'New Taipei City',
        '臺北市': 'Taipei City',
        '台北市': 'Taipei City', # alias of 臺北市
        '桃園市': 'Taoyuan City', # combined in 桃園縣
        '桃園縣': 'Taoyuan County',
        '臺中市': 'Taichung City',
        '台中市': 'Taichung City', # alias of 臺中市
        '臺南市': 'Tainan City',
        '台南市': 'Tainan City', # alias of 臺中市
        '高雄市': 'Kaohsiung City',
        '宜蘭縣': 'Yilan County',
        '新竹縣': 'Hsinchu County',
        '苗栗縣': 'Miaoli County',
        '彰化縣': 'Changhua County',
        '南投縣': 'Nantou County',
        '雲林縣': 'Yunlin County',
        '嘉義縣': 'Chiayi County',
        '屏東縣': 'Pingtung County',
        '臺東縣': 'Taitung County',
        '台東縣': 'Taitung County', # alias of 臺東縣
        '花蓮縣': 'Hualien County',
        '澎湖縣': 'Penghu County',
        '基隆市': 'Keelung City',
        '新竹市': 'Hsinchu City',
        '嘉義市': 'Chiayi City',
        '金門縣': 'Kinmen County',
        '連江縣': 'Lienchiang County'
        }

var_tribe_model = api.model('VarTribeModel',
                            {'area': fields.String(required=True, description='縣市', example='臺東縣'),
                             'amount': fields.Integer(required=True, description='人口變化數')
                            })

var_tribe_list_model = api.model('VarTribeListModel',
                                {'族別': fields.String(required=True, description='原住民族', example='布農族'),
                                 '新北市': fields.Integer(required=True, description='新北市人口變化數'),
                                 '臺北市': fields.Integer(required=True, description='臺北市人口變化數'),
                                 '桃園縣': fields.Integer(required=True, description='桃園縣人口變化數'), # 桃園市為省轄市，使用桃園縣
                                 '臺中市': fields.Integer(required=True, description='臺中市人口變化數'),
                                 '臺南市': fields.Integer(required=True, description='臺南市人口變化數'),
                                 '高雄市': fields.Integer(required=True, description='高雄市人口變化數'),
                                 '宜蘭縣': fields.Integer(required=True, description='宜蘭縣人口變化數'),
                                 '新竹縣': fields.Integer(required=True, description='新竹縣人口變化數'),
                                 '苗栗縣': fields.Integer(required=True, description='苗栗縣市人口變化數'),
                                 '彰化縣': fields.Integer(required=True, description='彰化縣人口變化數'),
                                 '南投縣': fields.Integer(required=True, description='南投縣人口變化數'),
                                 '雲林縣': fields.Integer(required=True, description='雲林縣人口變化數'),
                                 '嘉義縣': fields.Integer(required=True, description='嘉義縣人口變化數'),
                                 '屏東縣': fields.Integer(required=True, description='屏東縣人口變化數'),
                                 '臺東縣': fields.Integer(required=True, description='臺東縣人口變化數'),
                                 '花蓮縣': fields.Integer(required=True, description='花蓮縣人口變化數'),
                                 '澎湖縣': fields.Integer(required=True, description='澎湖縣人口變化數'),
                                 '基隆市': fields.Integer(required=True, description='基隆市人口變化數'),
                                 '新竹市': fields.Integer(required=True, description='新竹市人口變化數'),
                                 '嘉義市': fields.Integer(required=True, description='嘉義市人口變化數'),
                                 '金門縣': fields.Integer(required=True, description='金門縣人口變化數'),
                                 '連江縣': fields.Integer(required=True, description='連江縣人口變化數')
                                })

var_tribe_list_model_107 = api.model('VarTribeListModel107',
                                {'族別': fields.String(required=True, description='原住民族', example='布農族'),
                                 '新北市': fields.Integer(required=True, description='新北市人口變化數'),
                                 '臺北市': fields.Integer(required=True, description='臺北市人口變化數'),
                                 '桃園市': fields.Integer(required=True, description='桃園市人口變化數'), # 桃園市升格為直轄市
                                 '臺中市': fields.Integer(required=True, description='臺中市人口變化數'),
                                 '臺南市': fields.Integer(required=True, description='臺南市人口變化數'),
                                 '高雄市': fields.Integer(required=True, description='高雄市人口變化數'),
                                 '宜蘭縣': fields.Integer(required=True, description='宜蘭縣人口變化數'),
                                 '新竹縣': fields.Integer(required=True, description='新竹縣人口變化數'),
                                 '苗栗縣': fields.Integer(required=True, description='苗栗縣市人口變化數'),
                                 '彰化縣': fields.Integer(required=True, description='彰化縣人口變化數'),
                                 '南投縣': fields.Integer(required=True, description='南投縣人口變化數'),
                                 '雲林縣': fields.Integer(required=True, description='雲林縣人口變化數'),
                                 '嘉義縣': fields.Integer(required=True, description='嘉義縣人口變化數'),
                                 '屏東縣': fields.Integer(required=True, description='屏東縣人口變化數'),
                                 '臺東縣': fields.Integer(required=True, description='臺東縣人口變化數'),
                                 '花蓮縣': fields.Integer(required=True, description='花蓮縣人口變化數'),
                                 '澎湖縣': fields.Integer(required=True, description='澎湖縣人口變化數'),
                                 '基隆市': fields.Integer(required=True, description='基隆市人口變化數'),
                                 '新竹市': fields.Integer(required=True, description='新竹市人口變化數'),
                                 '嘉義市': fields.Integer(required=True, description='嘉義市人口變化數'),
                                 '金門縣': fields.Integer(required=True, description='金門縣人口變化數'),
                                 '連江縣': fields.Integer(required=True, description='連江縣人口變化數')
                                })

var_area_model = api.model('VarAreaModel',
                            {'tribe': fields.String(required=True, description='原住民族', example='布農族'),
                             'amount': fields.Integer(required=True, description='人口變化數')
                            })

var_area_list_model = api.model('VarAreaListModel',
                               {'區域別': fields.String(required=True, description='縣市', example='臺東縣'),
                                '總計': fields.Integer(required=True, description='人口變化數'),
                                '阿美族': fields.Integer(required=True, description='阿美族人口變化數'),
                                '泰雅族': fields.Integer(required=True, description='泰雅族人口變化數'),
                                '排灣族': fields.Integer(required=True, description='排灣族人口變化數'),
                                '布農族': fields.Integer(required=True, description='布農族人口變化數'),
                                '魯凱族': fields.Integer(required=True, description='魯凱族人口變化數'),
                                '卑南族': fields.Integer(required=True, description='卑南族人口變化數'),
                                '鄒族': fields.Integer(required=True, description='鄒族人口變化數'),
                                '賽夏族': fields.Integer(required=True, description='賽夏族人口變化數'),
                                '雅美族': fields.Integer(required=True, description='雅美族人口變化數'),
                                '邵族': fields.Integer(required=True, description='邵族人口變化數'),
                                '噶瑪蘭族': fields.Integer(required=True, description='噶瑪蘭族人口變化數'),
                                '太魯閣族': fields.Integer(required=True, description='太魯閣族人口變化數'),
                                '撒奇萊雅族': fields.Integer(required=True, description='撒奇萊雅族人口變化數'),
                                '賽德克族': fields.Integer(required=True, description='賽德克族人口變化數'),
                                '拉阿魯哇族': fields.Integer(required=True, description='拉阿魯哇族人口變化數'),
                                '卡那卡那富族': fields.Integer(required=True, description='卡那卡那富族人口變化數'),
                                '尚未申報': fields.Integer(required=True, description='尚未申報人口變化數')
                               })