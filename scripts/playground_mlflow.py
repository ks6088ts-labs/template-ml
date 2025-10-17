import logging
from typing import Annotated

import mlflow
import pandas as pd
import typer

# Import packages
from dotenv import load_dotenv
from mlflow.models import infer_signature  # type: ignore
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from template_ml.loggers import get_logger

app = typer.Typer(
    add_completion=False,
    help="Dash Playground CLI",
)

logger = get_logger(__name__)


def set_verbose_logging(
    verbose: bool,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)


@app.command(help="ref. https://mlflow.org/docs/latest/ml/tracking/quickstart/")
def quickstart(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-H",
            help="MLflow tracking server host",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-P",
            help="MLflow tracking server port",
        ),
    ] = 8080,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    set_verbose_logging(verbose)

    # ---
    # Train a model and prepare metadata for logging
    # ---

    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 8888,
    }

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # ---
    # Log the model and its metadata to MLflow
    # ---
    mlflow.set_tracking_uri(uri=f"http://{host}:{port}")
    mlflow.set_experiment("MLflow Quickstart")
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)

        # Infer the model signature
        signature = infer_signature(X_train, lr.predict(X_train))

        # Log the model, which inherits the parameters and metric
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            name="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )

        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_logged_model_tags(model_info.model_id, {"Training Info": "Basic LR model for iris data"})

        logger.info(f"model_info.model_uri: {model_info.model_uri}")


@app.command(
    help="ref. https://mlflow.org/docs/latest/ml/tracking/quickstart/#step-5---load-the-model-as-a-python-function-pyfunc-and-use-it-for-inference"
)
def inference(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-H",
            help="MLflow tracking server host",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-P",
            help="MLflow tracking server port",
        ),
    ] = 8080,
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id",
            "-M",
            help="Model ID to load from MLflow Model Registry",
        ),
    ] = "models:/m-xxx",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    set_verbose_logging(verbose)
    mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---
    # Load the model as a Python Function (pyfunc) and use it for inference
    # ---

    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_id)

    predictions = loaded_model.predict(X_test)

    iris_feature_names = datasets.load_iris().feature_names

    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    print(result[:4])


if __name__ == "__main__":
    assert load_dotenv(
        override=True,
        verbose=True,
    ), "Failed to load environment variables"
    app()
