"""
Created on Monday, Sep. 17, 2019

@author: whyang
"""
# -*- coding: utf-8 -*-
from flask import Flask, Blueprint
from flask_restplus import Api, Resource, fields
from environment.config import environment_config

###
# declare class initiating the server for restful API
# 
class Server(object):
    def __init__(self):
        self.app = Flask(__name__)
        ###
        # assign one blueprint to initiate the work of the enquery the population variation of indigenous peoples
        # the Blueprint's URL of app = /cip_pop_var
        #
        self.pop_var = Blueprint('cip_pop_var', __name__, url_prefix='/cip_pop_var')
        ###
        # declare the API for the blueprint of cip_pop_var
        # the base_URL = /cip_pop_var/1.0
        #
        self.api = Api(self.pop_var, 
                       version='1.0', 
                       prefix='/1.0', 
                       validate=True,
                       title='臺灣原住民族人口分佈數量變化資料集 API',
                       description='原住民族委員會 臺灣原民人口統計變動數據資料集 API',
                       doc=environment_config['swagger-url'])

    def run(self):
        self.app.register_blueprint(self.pop_var)
        self.app.run(debug=environment_config["debug"], port=environment_config["port"])
###
# initiate restful API server
#         
server = Server()

'''
#app = Flask(__name__)
self.app.register_blueprint(self.api_pop_var_100-102, url_prefix='/cip_pop_var/100-102')
self.app.register_blueprint(self.api_pop_var_100-103, url_prefix='/cip_pop_var/100-103')
self.app.register_blueprint(self.api_pop_var_100-107, url_prefix='/cip_pop_var/100-107')
self.app.register_blueprint(self.api_pop_var_102-103, url_prefix='/cip_pop_var/102-103')
self.app.register_blueprint(self.api_pop_var_102-107, url_prefix='/cip_pop_var/102-107')
self.app.register_blueprint(self.api_pop_var_103-107, url_prefix='/cip_pop_var/103-107')
'''
