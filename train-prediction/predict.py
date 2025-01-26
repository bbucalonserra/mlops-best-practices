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

# %%
