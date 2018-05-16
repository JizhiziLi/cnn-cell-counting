# CNN-CELL-COUNTING

* This is a well-structured Machine Learning microservice enables RESTful API.
* Uses to count the number of cells in a medical image based on CNN & other algorithms. 
* Relevant dissertation and demo slides can be viewed in `document` folder.
* Structure used for building in this case is [flask](http://flask.pocoo.org/).
* Any questions please contact __Jizhizi Li__ [jizhizili@gmail.com]
* Relevant docker repository is [here](https://hub.docker.com/r/jizhizili/cnn-cell-counting/)

## How to Setup

**Step 1:** Create a new conda virtual env. - `conda create -n $name python=3.6`

**Step 2:** Go into virtual env. - `source activate $name`

**Step 3:** Install all the required packages - `pip install -r requirements`

**Step 4:** Install matplotlib seperately  - `conda install matplotlib`

**Step 5:** Deactivate the env - `source deactivate $name`

## How to Run App

**Step 1:** Run the virtual env `Source activate $name`

**Step 2:** Run `python run.py` to start


## Routes

1.  POST `/print_coordinate`:

    ```json
    {
        "num":3
    }
    ```

    Give a specific number of image you want to process, get the well with coordinate printed in save folder.

2.  POST `/plot_data`:

    ```json
    {
        "plot_name":"train_plot"
    }
    ```

    Give a pkl name, plots data and label on image.

3.  POST `/generate_dataset`:

    ```json
    {
        "cell_size": 5,
        "crop_length": 50,
        "crop_width": 50,
        "crops_number_per_image": 60,
        "images_range": [
            1,
            3
        ],
        "save_name": "train"
    }
    ```

    Pass some parameters to generate a balanced labelled dataset used for training and testing.

4.  POST `/linear_model`:

    ```json
    {
        "test": {
            "choice": 1,
            "data_set": "test"
        },
        "train": {
            "choice": 1,
            "data_set": "train"
        }
    }
    ```

    Pass some parameters to train linear model on training set and test on testing set. Relevant logs and figs will be generated.

5.  POST `/logistic_model`:

    ```json
    {
        "test": {
            "choice": 1,
            "data_set": "test"
        },
        "train": {
            "choice": 2,
            "data_set": "train"
        }
    }
    ```

    Pass some parameters to train logistic model on training set and test on testing set. Relevant logs and figs will be generated.

6.  POST `/cnn_model/train`:

    ```json
    {
        "cell_number": 16,
        "choice": "linear_count",
         "height": 50,
        "learning_rate": 0.0001,
        "number": 1000,
        "patience": 5000,
        "train_set_file": "train",
        "width": 50
    }
    ```

    Pass some parameters to train cnn model on training set. Other parameters include `patience` and `learning_rate` can also be modified to increase training efficiency and model performance.