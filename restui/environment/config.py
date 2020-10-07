"""
Created on Monday, Sep. 17, 2019

@author: whyang
"""
# -*- coding: utf-8 -*-
import os

env = os.environ.get("PYTHON_ENV", "development")
port = os.environ.get("PORT", 8080)

all_environments = {
    "development": { "port": 5000, "debug": True, "swagger-url": "/api" },
    "production": { "port": port, "debug": False, "swagger-url": None  }
}

environment_config = all_environments[env]
