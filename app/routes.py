from app import app, module
from flask import Flask, request



# @app.route('/')
@app.route('/index')
def index():
    return 'Hello, World!'


#####
# route /print_coordinate is used to process an image and print its coordinate for a well
# 
# eg. curl -H "Content-Type: application/json" -X POST -d '{"num":5}' "127.0.0.1:5000/print_coordinate"
#####
@app.route('/print_coordinate', methods=['POST'])
def print_coordinate():
    try:
        body = request.get_json()
        image_number = body['num']
        module._print_coordinate(image_number)
        return 'Print Coordinate'
    except Exception as e:
        print(str(e))
        return 'We got Error'