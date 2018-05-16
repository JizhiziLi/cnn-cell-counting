#!/bin/bash
# Script to setup a standard machine learning model based on flask.
# Copyright (C) 2018 Jizhizi Li - All Rights Reserved
# @author Jizhiz Li
# use `chmod +x setup.sh` to set permission

# Only run this scirpt first time when you download this repository.
# It will help you setup virutal env and install all required packages.


#create the virtual machine used here.
echo "Please give a name for the virtual env you want to work on:"
read virtual_name
conda create -n $virtual_name python=3.6
echo "Now installing all packages..."
pip install -r requirements.txt
conda install matplotlib