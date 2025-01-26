# MLOps Best Practices
In this project, we'll create a machine learning model and apply best practices on it, including:
  * ML Flow to manage the life cycle of the models and track its results
  * Unity tests
  * 

## 1. Creating ML Flow Environment
The first step is to download Python packages and create environements with the specific configurations (e.g. Python 3.13) and libraries in specific versions. This step is really important to perform analysis and create model in a controled environment. For the project, we'll use [Anadonda](https://www.anaconda.com/), an open ecosystem for sourcing, building, and deploying data science and AI initiatives.


### 1.1 Create Server in Anaconda
In the first section, we'll create the ML Flow environment. For this, let's open the [Anaconda](https://www.anaconda.com/) terminal, find the repository we want to create the environment using `cd` and type the command `conda create --name mlflow-server python 3.13`. In this case, we'll use Python version 3.13.   Then, just type `y` (yes) to confirm the creation of the environment.  

![Creating Environment in Anaconda](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/create-environment.png)  
*Figure 1: Creating Environment in Anaconda*  

![Creating Environment in Anaconda](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/create-environment.png)  
*Figure 2: Creating Environment in Anaconda*


### 1.2 Activate Server
The next step is simply activate the server by using ´conda activate mlflow-server´. To deactivate, simply type `conda deactivate mlflow-server`.
![Activate Server in Anaconda](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/activate-server.png)  
*Figure 3: Activate Server in Anaconda*


### 1.3 Install ML Flow Library
Install the ML Flow in the environment by using `pip install mlflow`
![Install ML Flow in the Environment](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/pip-install-mlflow.png)  
*Figure 4: Install ML Flow in the Environment*


In order to see all command for MLFlow, type `mlflow`.
![Check Commands for ML Flow](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/mlflow-commands.png)  
*Figure 5: Check Commands for ML Flow*


### 1.4 Server Port
It's needed to find the port of the server in order to open the ML Flow in the browser, for this, type `mlflor server`. It's really important to keep the port open! Otherwise it's not possible to use the ML Flow. To open the ML Flow, just type the port into the browser and open the ML Flow
![Find Server Port](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/server-port.png)  
*Figure 6: Find Server Port*


## 2. Creating Project Environment
Now, we must create an environment for the churn project itself. We'll create a environment call `churn-server` using Python=3.13 (just the same way we did in the previous step), than, activate it.
In addition, we must install all libraries in this environment, in our case, we'll install: pandas, sklearn and mlflow:

![Creating Project Environment](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/churn-project-environment.PNG)  
*Figure 7: Creating Project Environment*

Thus, we must install the libraries we'll use in the environment. We must: `pip install pandas scikit-learn mlflow`:
![Installing Libraries](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/pip-install-libraries-environment.PNG)  
*Figure 7: Installing Libraries*

**NOTE: this is the environment that the executer (KERNEL) will use to execute the python scripts.**

## 3. Setting ML Flow
Type the port into the browser and open the ML Flow. A model is associated with multiple experiments, and each experiment consists of several runs. A run represents a single execution with a specific set of parameters or configurations (e.g., testing different hyperparameters). Each run generates a distinct result, providing insights into the model's performance under those conditions. Once the runs are completed, the best-performing ones are selected for logging in the model. For example, if in Experiment 1, Run 2 yields the best results, you would log Run 2 as part of the model's versioning or final configuration. Check the flowchart below and the models and experiments screen:

### 3.1 General ML Flow

![Model](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/mlflow-flow-chart.PNG)  
*Figure 8: Flow Chart of ML Flow*


![Model](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/ml-flow-models.png)  
*Figure 9: Model*


![Experiments](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/ml-flow-experiments.png)  
*Figure 10: Experiments*


### 3.2 Creating Model
We'll create our model for the project.
![Create Model](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/creating-model.PNG)  
*Figure 11: Create Model*


### 3.3 Creating Experiment
Creating an experiment for our project.
![Create Model](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/churn-experiment-created.PNG)  
*Figure 12: Create Experiment*


Now, the ML Flow is properly set.
![ML Flow Set](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/churn-model-created.PNG)  
*Figure 13: ML Flow Set*


## 4 Code Adjustment
There are a few code lines we should add in order to connect the script with ML Flow. The first one is to add the port (found in previous steps) and the experiment ID also created inthe previous step. The experiment ID can be found be clicking in the experiment information:


![Finding Experiment ID](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/experiment-id.png)  
*Figure 14:Finding Experiment ID*


The code the must be add can be found below, right after importing the libraries:
```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_id=966213590755579700)
```










