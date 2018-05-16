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


#####
# route /generate_dataset is used to process an image and print its coordinate for a well
# 
# eg. curl -H "Content-Type: application/json" -X POST -d 
# "{
#  "cell_size": 5,
#  "crop_length": 50,
#  "crop_width": 50,
#  "crops_number_per_image": 60,
#  "images_range": [1, 3],
#  "save_name": "train"
# }"
# to "127.0.0.1:5000/generate_dataset"
#####
@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    try:
        body = request.get_json()
        module._generate_dataset(body)
        return 'Generate dataset and save to folder params.'
    except Exception as e:
        print(str(e))
        return 'We got Error'


#####
# route /plot_data is used to plot split images based on pkl data
# 
# eg. curl -H "Content-Type: application/json" -X POST -d '{"plot_name":"train_plot"}' "127.0.0.1:5000/plot_data"
#####
@app.route('/plot_data', methods=['POST'])
def plot_data():
    try:
        body = request.get_json()
        plot_name = body['plot_name']
        module._plot_data_and_label(plot_name)
        return 'Plot Data'
    except Exception as e:
        print(str(e))
        return 'We got Error'




#####
# route /logistic_model is used to run logistic regression model on data
# train_set is the name of dataset used for training
# choice is the approach chosen in logistic model
# eg. curl -H "Content-Type: application/json" -X POST -d 
# "{
#	"train":{"choice":1,"data_set":"train"},
#	"test":{"choice":1,"data_set":"train"}
# "}
# to "127.0.0.1:5000/logistic_model"
#####
@app.route('/logistic_model', methods=['POST'])
def logistic_model():
    try:
        body = request.get_json()
        module._logistic_model(body)
        return 'Logistic model has been run.'
    except Exception as e:
        print(f'Error here: {str(e)}')
        return 'We got Error'




#####
# route /linear_model is used to run linear regression model on data
# train_set is the name of dataset used for training
# choice is the approach chosen in linear model
# eg. curl -H "Content-Type: application/json" -X POST -d 
# "{
#	"train":{"choice":1,"data_set":"train"},
#	"test":{"choice":1,"data_set":"train"}
# "}
# to "127.0.0.1:5000/linear_model"
#####
@app.route('/linear_model', methods=['POST'])
def linear_model():
    try:
        body = request.get_json()
        module._linear_model(body)
        return 'Linear model has been run.'
    except Exception as e:
        print(f'Error here: {str(e)}')
        return 'We got Error'



#####
# route /cnn_model_train is used to train cnn model on data
# eg. curl -H "Content-Type: application/json" -X POST -d 
# "{
#  "width": 50,
#  "height": 50,
#  "patience": 5000,
#  "choice": "linear_count",
#  "number": 1000,
#  "cell_number": 16,
#  "learning_rate": 0.0001,
#  "train_set_file": "train"
# }"
# to "127.0.0.1:5000/cnn_model/train"
#####
@app.route('/cnn_model/train', methods=['POST'])
def cnn_model_train():
    try:
        body = request.get_json()
        module._cnn_model_train(body)
        return 'CNN model has been run.'
    except Exception as e:
        print(f'Error here: {str(e)}')
        return 'We got Error'