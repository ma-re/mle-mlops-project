import os
import argparse

import pandas as pd

import mlflow
from mlflow.tracking.client import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Parse input 
parser = argparse.ArgumentParser()
parser.add_argument(
    '--cml_run', default=False, action=argparse.BooleanOptionalAction, required=True
)
args = parser.parse_args()
cml_run = args.cml_run


# Set google application credentials
GOOGLE_APPLICATION_CREDENTIALS = "./credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Set MLFlow tracking URI and connection to MLFlow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('mlops-pipeline-project')
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# Set variables
color = 'green'
year = '2022'
month = '1'
features = ['PULocationID', 'DOLocationID', 'trip_distance', 'fare_amount', 'total_amount', 'passenger_count']
target = 'trip_duration_minutes'
model_name = f'{color}-taxi-project-model'

## Download the data (not needed any more)
#if not os.path.exists(f"./data/{color}_tripdata_{year}-{month:02d}.parquet"):
#    os.system(f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet")

# Load the data
df = pd.read_parquet(f"data/green_tripdata_2022-01.parquet")


# calculate the trip duration in minutes and drop trips that are less than 1 minute and more than 1 hour
def calculate_trip_duration_in_minutes(df):
    df["trip_duration_minutes"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["trip_duration_minutes"] >= 1) & (df["trip_duration_minutes"] <= 60)]
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 8)]
    df = df[features + [target]]
    return df


df_processed = calculate_trip_duration_in_minutes(df)

y=df_processed["trip_duration_minutes"]
X=df_processed.drop(columns=["trip_duration_minutes"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Model training and tracking
with mlflow.start_run():

    # Set tags to track
    tags = {
        "model": "linear regression",
        "developer": "Mia",
        "dataset": f"{color}-taxi",
        "year": year,
        "month": month,
        "features": features,
        "target": target
    }
    mlflow.set_tags(tags)
    
    # Train model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predict on train and test set
    y_pred_train = lr.predict(X_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    y_pred_test = lr.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    
    # Log metrics
    metrics = {'Train RMSE': rmse_train, 'Test RMSE': rmse_test}
    mlflow.log_metrics(metrics)

    # Log model
    mlflow.sklearn.log_model(lr, 'model')
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/sklearn-model"
    model_name = f"{color}-taxi-ride-duration-model"
    mlflow.register_model(model_uri=model_uri, name=model_name)

    # Track model details
    model_version = 1
    new_stage = "Production"
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=False
    )

if cml_run:
    with open('metrics.txt', 'w') as f:
        f.write(f'RMSE on the Train Set: {rmse_train}')
        f.write(f'RMSE on the Test Set: {rmse_test}')