# CNN-CELL-COUNTING

* This is a well-structured microservice to detect number of cells in image based on CNN & other algorithms. 
* Related dissertation and demo slides can be viewed in `document` folder.
* Structure used for RESTful methods in this case is flask[http://flask.pocoo.org/].
* Any questions please contact *JizhiziLi* [jizhizili@gmail.com]

## How to Setup

**Step 1:** Create a new conda virtual env. - `conda create -n $name python=3.6`

**Step 2:** Go into virtual env. - `source activate $name`

**Step 3:** Install all the required packages - `pip install -r requirements`

**Step 4:** Deactivate the env - `source deactivate $name`

## How to Run App

1. Run the virtual env `Source activate $name`
2. Run `python run.py` to start


## Routes

*1* POST `/print_coordinate`:
    Give a specific of the number of image you want to process, get coordinate printed well in save folder.
