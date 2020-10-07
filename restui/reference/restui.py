"""
Created on Monday, Sep. 17, 2019

@author: whyang
"""
# -*- coding: utf-8 -*-
from flask import Flask, Blueprint
from flask_restplus import Api, Resource, fields

api_pop_var = Blueprint('popvar_api', 
                   __name__, 
                   url_prefix='/cip_pop_var/100-102')

api = Api(api_pop_var, 
          version='0.1',
          #prefix='/100-102',
          title='CIP_Population Variation Statistics (100-102) API', 
          description='原住民族委員會 原民人口變異統計數據(100年-102年) API')

ns = api.namespace('tribe', 
                   description='原住民族人口變化(100-102):族')

TRIBES = {
    '阿美族': {'tribe1': 'Ah Mei'},
    '泰雅族': {'tribe2': 'Tai Yeah'}
    }

var_tribe_model = api.model('VarTribeModel',
                            {#'tribe': fields.String(required=True, description='原住民族'),
                             'area': fields.String(required=True, description='縣市'),
                             'amount': fields.Integer(required=True, description='人口變化數')
                             })

var_tribe_list_model = api.model('VarTribeListModel', 
                        {'area': fields.List(fields.Nested(var_tribe_model)),
                         #'total': fields.Integern='原民人口變化數')
                         })

def abort_if_tribe_doesnt_exist(tribe):
    if tribe not in TRIBES:
        api.abort(404, "tribe {} doesn't exist".format(tribe))

'''
parser = api.parser()
parser.add_argument('tribe', type=str, required=True, help='The name of indigenous peoples', location='form')
'''

@ns.route('/<string:tribe>')
@api.doc(responses={404: 'Todo not found'}, params={'tribe': "The tribe's name"})
class Tribe(Resource):
    '''Show a single todo item and lets you delete them'''
    @api.doc(description='tribe should be in {0}'.format(', '.join(TRIBES.keys())))
    @api.marshal_with(var_tribe_list_model)
    def get(self, tribe):
        '''Fetch a given resource'''
        abort_if_tribe_doesnt_exist(tribe)
        return TRIBES[tribe]

    '''
    @api.doc(responses={204: 'Todo deleted'})
    def delete(self, todo_id):
        # Delete a given resource
        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        return '', 204

    @api.doc(parser=parser)
    @api.marshal_with(var_tribe_list_model)
    def put(self, tribe):
        #Update a given resource
        args = parser.parse_args()
        task = {'task': args['task']}
        TODOS[var_tribe_list_model] = task
        return task
    '''

@ns.route('/')
class TribeList(Resource):
    '''Shows a list of all todos, and lets you POST to add new tasks'''
    @api.marshal_list_with(var_tribe_list_model)
    def get(self):
        '''List all todos'''
        return [{'tribe': tribeName, 'description': tribeDes} for tribeName, tribeDes in TRIBES.items()]

    '''
    @api.doc(parser=parser)
    @api.marshal_with(var_tribe_list_model, code=201)
    def post(self):
        #Create a todo
        args = parser.parse_args()
        todo_id = 'todo%d' % (len(TODOS) + 1)
        TODOS[todo_id] = {'task': args['task']}
        return TODOS[todo_id], 201
    '''

@ns.route('/<string:tribe>')
@api.doc(responses={404: 'Todo not found'}, params={'tribe': "The tribe's name"})
class Tribe(Resource):
    '''Show a single todo item and lets you delete them'''
    @api.doc(description='tribe should be in {0}'.format(', '.join(TRIBES.keys())))
    @api.marshal_with(var_tribe_list_model)
    def get(self, tribe):
        '''Fetch a given resource'''
        abort_if_tribe_doesnt_exist(tribe)
        return TRIBES[tribe]

    '''
    @api.doc(responses={204: 'Todo deleted'})
    def delete(self, todo_id):
        # Delete a given resource
        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        return '', 204

    @api.doc(parser=parser)
    @api.marshal_with(var_tribe_list_model)
    def put(self, tribe):
        #Update a given resource
        args = parser.parse_args()
        task = {'task': args['task']}
        TODOS[var_tribe_list_model] = task
        return task
    '''
if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(api_pop_var)
    app.run(debug=True)