"""
Created on Monday, Sep. 17, 2019

@author: whyang
"""
# -*- coding: utf-8 -*-
from server.restAPI import server

# Need to import all resources
from resources.popvar_resource_100_102 import *
from resources.popvar_resource_100_103 import *
from resources.popvar_resource_100_107 import *
from resources.popvar_resource_102_103 import *
from resources.popvar_resource_102_107 import *
from resources.popvar_resource_103_107 import *

if __name__ == '__main__':
    server.run()