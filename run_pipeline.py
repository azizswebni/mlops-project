from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    train_pipeline()

    print(
        f"mlflow UI command ==> mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
    )
