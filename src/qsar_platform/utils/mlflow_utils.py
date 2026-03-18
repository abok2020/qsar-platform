import mlflow


def start_run(run_name: str):
    mlflow.set_experiment("qsar-platform")
    return mlflow.start_run(run_name=run_name)


def log_common_params(model_name: str, n_splits: int, input_path: str, output_path: str):
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("n_splits", n_splits)
    mlflow.log_param("input_path", input_path)
    mlflow.log_param("output_path", output_path)


def log_metric(name: str, value: float):
    mlflow.log_metric(name, float(value))


def log_artifact(path: str):
    mlflow.log_artifact(path)
