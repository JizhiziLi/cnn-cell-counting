# CNN-CELL-COUNTING

* This is a well-structured Machine Learning microservice enables RESTful API.
* Uses to count the number of cells in a medical image based on CNN & other algorithms. 
* Relevant dissertation and demo slides can be viewed in `document` folder.
* Structure used for building in this case is flask [http://flask.pocoo.org/].
* Any questions please contact __Jizhizi Li__ [jizhizili@gmail.com]

## How to Setup

**Step 1:** Create a new conda virtual env. - `conda create -n $name python=3.6`

**Step 2:** Go into virtual env. - `source activate $name`

**Step 3:** Install all the required packages - `pip install -r requirements`

**Step 4:** Deactivate the env - `source deactivate $name`

## How to Run App

**Step 1:** Run the virtual env `Source activate $name`

**Step 2:** Run `python run.py` to start


## Routes

1.  POST `/print_coordinate`:

    Give a specific of the number of image you want to process, get coordinate printed well in save folder.
2.  POST `/generate_dataset`:

    Pass some parameters to generate a balanced labelled dataset used for training and testing.
3.  POST `/linear_model`:

    Pass some parameters to train linear model on training set and test on testing set. Relevant logs and figs will be generated.
4.  POST `/logistic_model`:

    Pass some parameters to train logistic model on training set and test on testing set. Relevant logs and figs will be generated.
