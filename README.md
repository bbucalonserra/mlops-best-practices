# MLOps Best Practices
In this project, we'll create a machine learning model and apply best practices on it, including:
  * ML Flow to manage the life cycle of the models and track its results
  * Unity tests
  * 

## 1. Creating Environments
The first step is to download Python packages and create environements with the specific configurations (e.g. Python 3.13) and libraries in specific versions. This step is really important to perform analysis and create model in a controled environment. For the project, we'll use [Anadonda](https://www.anaconda.com/), an open ecosystem for sourcing, building, and deploying data science and AI initiatives.


### 1.1 ML Flow
In the first section, we'll create the ML Flow environment. For this, let's open the [Anadonda](https://www.anaconda.com/) terminal, find the reposity we want to create the environment using `cd` and type the command `conda create --name mlflow-server python 3.13`, in this case, we'll use Python version 3.13. Than, just type y (yes) to confirm the creation of the environment.



The next step is simply activate the environment by using ´conda activate mlflow-server´. To deactivate, simply type `conda deactivate mlflow-server`.



Instal the ML Flow in the environment by using `pip install mlflow`



In order to see all command for MLFlow, type `mlflow`.



It's needed to find the port of the server in order to open the ML Flow in the browser, for this, type `mlflor server`.



IT's really important to keep the port open! Otherwise it's not possible to use the ML Flow.



Type the port into the browser and open the ML Flow. A model is associated with multiple experiments, and each experiment consists of several runs. A run represents a single execution with a specific set of parameters or configurations (e.g., testing different hyperparameters). Each run generates a distinct result, providing insights into the model's performance under those conditions. Once the runs are completed, the best-performing ones are selected for logging in the model. For example, if in Experiment 1, Run 2 yields the best results, you would log Run 2 as part of the model's versioning or final configuration.



Creating a model for our project



Creating an experiment for our project



Now, the ML Flow is properly set.










