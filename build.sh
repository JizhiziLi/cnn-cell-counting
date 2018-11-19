#!/bin/bash
# Script to setup a standard machine learning model based on flask.
# Copyright (C) 2018 Jizhizi Li - All Rights Reserved
# @author Jizhiz Li
# use `chmod +x build.sh` to set permission

# Only run this scirpt once when you want to build a similar void project.
# If you clone the repo directly, please use setup.sh instead.

echo "Hello there, I'll help you set up a basic flask structure for your machine learning model"
echo "Please name the project you want to build:"
read project_name
mkdir $project_name
cd $project_name
touch requirements.txt
touch README.md
echo "Flask
connexion
Flask-Injector
fastavro
pytest
" > requirements.txt
touch README.md

echo "#  $project_name
* Machine learning model structure based on lightweight framework - flask

## How to Setup

**Step 1:** Navigate to the python virtual env.

**Step 2:** Run python run.py

**Step 3:** Install the Application

## How to Run App

1. cd to the repo
2. Run `python run.py`" > README.md

#create the virtual machine used here.
echo "And the name of virtual machine you want to build:"
read virtual_name
conda create -n $virtual_name python=3.6
source activate $virtual_name
pip install -r requirements.txt
source deactivate $virtual_name


#create the root run.py file
touch run.py
echo "from app import app
app.run(debug=True)" > run.py


touch .gitignore
echo "# IDEA config files #
.idea
.idea/*

# PYTHON running file #
*.pyc
*.pyo

# test images #
*.jpg
*.png
*.jpeg

# vscode env #
.vscode" > .gitignore

touch config.py
echo "# This file is used to store all the config variables used in project."

mkdir app
cd app
mkdir module
mkdir static
mkdir templates
mkdir tests
touch __init__.py
touch routes.py
echo "from flask import Flask

app = Flask(__name__)

from app import routes


app.run(debug=True)" > __init__.py

echo "from app import app

@app.route('/')
@app.route('/index')
def index():
    return 'Hello, World!'" > routes.py

cd module
touch __init__.py
echo "from .model import *" > __init__.py
touch model.py
echo "# ML model can be written here" > model.py

echo "You can use
					source activate $virtual_name
					and
					python run.py
to run the project now!"
