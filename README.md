# ML Operations - Project Life Cycle
In this project, we'll create a quick machine learning model and apply ML Operations best practices on it, including creating an environment for Python, ML Flow environment to manage the life cycle of the experiments and track its results and unity tests using PyTest.

**NOTE: some best practices, such as usage of Cloud Computing, Databricks flow, clusters and Apache Spark framework (instead of Pandas API) is not going to be shown here, since this is a local project.**

# Table of Contents
- [1. Creating ML Flow Environment](#1-creating-ml-flow-environment)
  - [1.1 Create Server in Anaconda](#11-create-server-in-anaconda)
  - [1.2 Activate Server](#12-activate-server)
  - [1.3 Install ML Flow Library](#13-install-ml-flow-library)
  - [1.4 Server Port](#14-server-port)
- [2. Creating Project Environment](#2-creating-project-environment)
- [3. Setting ML Flow](#3-setting-ml-flow)
  - [3.1 General ML Flow](#31-general-ml-flow)
  - [3.2 Creating Model](#32-creating-model)
  - [3.3 Creating Experiment](#33-creating-experiment)
- [4. Code Adjustment](#4-code-adjustment)
  - [4.1 Port and Experiment ID](#41-port-and-experiment-id)
  - [4.2 Autolog and Metrics](#42-autolog-and-metrics)
  - [4.3 Final Code](#43-final-code)
- [5. Script and ML Flow](#5-script-and-ml-flow)
  - [5.1 Runs in Experiment](#51-runs-in-experiment)
  - [5.2 Registering Experiment in a Model](#52-registering-experiment-in-a-model)
- [6. Productizing the Model](#6-productizing-the-model)
- [7. Unity Test](#7-unity-test)

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
*Figure 8: Installing Libraries*

**NOTE: this is the environment that the executer (KERNEL) will use to execute the python scripts.**

## 3. Setting ML Flow
Type the port into the browser and open the ML Flow. A model is associated with multiple experiments, and each experiment consists of several runs. A run represents a single execution with a specific set of parameters or configurations (e.g., testing different hyperparameters). Each run generates a distinct result, providing insights into the model's performance under those conditions. Once the runs are completed, the best-performing ones are selected for logging in the model. For example, if in Experiment 1, Run 2 yields the best results, you would log Run 2 as part of the model's versioning or final configuration. Check the flowchart below and the models and experiments screen:

### 3.1 General ML Flow

![Model](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/mlflow-flow-chart.PNG)  
*Figure 9: Flow Chart of ML Flow*


![Model](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/ml-flow-models.png)  
*Figure 10: Model*


![Experiments](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/ml-flow-experiments.png)  
*Figure 11: Experiments*


### 3.2 Creating Model
We'll create our model for the project.
![Create Model](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/creating-model.PNG)  
*Figure 12: Create Model*


### 3.3 Creating Experiment
Creating an experiment for our project.
![Create Model](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/churn-experiment-created.PNG)  
*Figure 13: Create Experiment*


Now, the ML Flow is properly set.
![ML Flow Set](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/churn-model-created.PNG)  
*Figure 13: ML Flow Set*


## 4 Code Adjustment
There are a few code lines we should add in order to connect the script with ML Flow. The first one is to add the port (found in previous steps) and the experiment ID also created inthe previous step. The experiment ID can be found be clicking in the experiment information:


![Finding Experiment ID](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/experiment-id.png)  
*Figure 14: Finding Experiment ID*


### 4.1 Port and Experiment ID
The code the must be add can be found below, right after importing the libraries:
```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_id=243927022239586130)
```


### 4.2 Autolog and Metrics
We also need to write our trainning script inside the function below:
```python
with mlflow.start_run():

    mlflow.sklearn.autolog()

    (...)

mlflow.log_metrics({"acc_train": acc_train, "acc_test": acc_test})
```

This part of the code is responsible to collect the trainning information and the metrics in order to store in ML Flow.

### 4.3 Final Code
The final code for the train file in [THIS SCRIPT](https://github.com/bbucalonserra/mlops-best-practices/blob/main/train-prediction/train.py) is: 

```python
# %%
import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
import mlflow

# %%
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_id=243927022239586130)

# %%
df = pd.read_csv("data/abt.csv", sep=",")

features = df.columns[2:-1]
target = "churn_flag"

X = df[features]
y = df[target]

df.head()
# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    test_size=0.2,
                                                                    random_state=42)

print("Train:", y_train.mean())
print("Test:", y_test.mean())

# %%

with mlflow.start_run():

    mlflow.sklearn.autolog()

    # clf = tree.DecisionTreeClassifier(min_samples_leaf=100,
    #                                   min_samples_split=10,
    #                                   random_state=42)

    clf = ensemble.RandomForestClassifier(n_estimators=500,
                                          min_samples_leaf=35,
                                          random_state=42)
    clf.fit(X_train, y_train)

    y_train_predict = clf.predict(X_train)
    y_test_predict = clf.predict(X_test)

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    acc_test = metrics.accuracy_score(y_test, y_test_predict)

    mlflow.log_metrics({"acc_train": acc_train, "acc_test": acc_test})


# %%
print("Acurácia train:", acc_train)
print("Acurácia test:", acc_test)
# %%
```



## 5 Script and ML Flow
Now, everytime the script is ran, the results will be stored in the experiments. It's important now to remanage this page. So, we'll remove some columns we're not using and leave only the ones for comparisson of different runs. Let's put `acc_test, acc_train, min_sample_leafs, min_samples_sp`. Now, wan can run with different paramethers and check their results in a single page, including the duration of the algorithm (in order to checkc performance).

### 5.1 Runs in Experiment
Let's check the runs and what's inside of it.

![Runs](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/experiment-runs.PNG)
*Figure 15: Runs*

By clicking in a run, **we can check all informations about the trained experiment, including the artifacts and the model in binary.**

![Inside Runs](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/inside-run.png)
*Figure 16: Inside Runs*

![Inside Runs](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/experiment-artifacts.png)
*Figure 17: Inside Runs*

### 5.2 Registering Experiment in a Model
To complete, you can choose the best run and click in the best run and register it in the model.

![Registering Model](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/register-model.png)
*Figure 18: Registering Model*

![Registered Model in Models Page](https://github.com/bbucalonserra/mlops-best-practices/blob/main/images/experiment-registered-in-model.PNG)
*Figure 19: Registered Model in Models Page*


## 6. Productizing the Model
One way to productize the model is to create a script that collects the last experiment registered in the models. This is done by using [THIS SCRIPT](https://github.com/bbucalonserra/mlops-best-practices/blob/main/train-prediction/predict.py).

```python
# %%
import mlflow.client
import pandas as pd

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# %%
client = mlflow.client.MlflowClient()
version = max([int(i.version) for i in client.get_latest_versions("churn-model")])

# %%
model = mlflow.sklearn.load_model(f"models:/churn-model/{version}")

# %%
df = pd.read_csv("data/abt.csv", sep=",")
df

# %%
X = df.head()[model.feature_names_in_]
proba = model.predict_proba(X)
proba
```

## 7. Unity Test and Red-Green-Refactor
Unit tests are additional pieces of code designed to test specific components or functions of the main code in isolation, allowing developers to verify their behavior and ensure they work as expected. For this, we'll use **Pytest**.

The process RGR (Red-green-refactor) should be applied. This cycle assures that the code is ready to deploy, tested and well projected. It's divided in three parts:
  1. Red - Write a test that fails, before writting the main code, the idea is ensure that the test is correct the the pipeline / model does not exist
  2. Green - Implement a test that passes, it should be easy and not worring about best practices
  3. Refactor - Rewrite the code, but now, making it cleaner, more understandable and efficient

Applying unity test in order to check if the received dataset has all columns. Check the file [HERE](https://github.com/bbucalonserra/mlops-best-practices/blob/main/test/test_unitary.py).


import pandas as pd
from sklearn import model_selection

```python
def test_expected_columns_function():
    df = pd.read_csv("data/abt.csv", sep=",")
    expected_columns = [
    "dt_ref", "id_customer", 
    "days_first_iteration_life", "days_last_iteration_life", "qty_iterations_life", 
    "current_balance_life", "points_accumulated_life", "negative_points_life", 
    "frequency_life", "points_accumulated_per_day_life", "qty_messages_life", 
    "qty_redemptions_ponies_life", "messages_per_day_life", "pct_transaction_day01_life", 
    "pct_transaction_day02_life", "pct_transaction_day03_life", "pct_transaction_day04_life", 
    "pct_transaction_day05_life", "pct_transaction_day06_life", "pct_transaction_day07_life", 
    "pct_transaction_morning_life", "pct_transaction_afternoon_life", "pct_transaction_night_life", 
    "avg_days_recurrence_life", "median_days_recurrence_life", 
    "days_first_iteration_d7", "days_last_iteration_d7", "qty_iterations_d7", 
    "current_balance_d7", "points_accumulated_d7", "negative_points_d7", 
    "frequency_d7", "points_accumulated_per_day_d7", "qty_messages_d7", 
    "qty_redemptions_ponies_d7", "messages_per_day_d7", "pct_transaction_day01_d7", 
    "pct_transaction_day02_d7", "pct_transaction_day03_d7", "pct_transaction_day04_d7", 
    "pct_transaction_day05_d7", "pct_transaction_day06_d7", "pct_transaction_day07_d7", 
    "pct_transaction_morning_d7", "pct_transaction_afternoon_d7", "pct_transaction_night_d7", 
    "avg_days_recurrence_d7", "median_days_recurrence_d7", 
    "days_first_iteration_d14", "days_last_iteration_d14", "qty_iterations_d14", 
    "current_balance_d14", "points_accumulated_d14", "negative_points_d14", 
    "frequency_d14", "points_accumulated_per_day_d14", "qty_messages_d14", 
    "qty_redemptions_ponies_d14", "messages_per_day_d14", "pct_transaction_day01_d14", 
    "pct_transaction_day02_d14", "pct_transaction_day03_d14", "pct_transaction_day04_d14", 
    "pct_transaction_day05_d14", "pct_transaction_day06_d14", "pct_transaction_day07_d14", 
    "pct_transaction_morning_d14", "pct_transaction_afternoon_d14", "pct_transaction_night_d14", 
    "avg_days_recurrence_d14", "median_days_recurrence_d14", 
    "days_first_iteration_d28", "days_last_iteration_d28", "qty_iterations_d28", 
    "current_balance_d28", "points_accumulated_d28", "negative_points_d28", 
    "frequency_d28", "points_accumulated_per_day_d28", "qty_messages_d28", 
    "qty_redemptions_ponies_d28", "messages_per_day_d28", "pct_transaction_day01_d28", 
    "pct_transaction_day02_d28", "pct_transaction_day03_d28", "pct_transaction_day04_d28", 
    "pct_transaction_day05_d28", "pct_transaction_day06_d28", "pct_transaction_day07_d28", 
    "pct_transaction_morning_d28", "pct_transaction_afternoon_d28", "pct_transaction_night_d28", 
    "avg_days_recurrence_d28", "median_days_recurrence_d28", 
    "churn_flag"
]
    for col in expected_columns:
        assert col in df.columns, f"The expected column {col} isn't found in the dataset."
```
