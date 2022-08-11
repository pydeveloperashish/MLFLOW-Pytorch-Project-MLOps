import mlflow



if __name__ == '__main__':
    with mlflow.start_run(run_name = "main") as runs:
        mlflow.run(".", "stage_01", use_conda = False)
        mlflow.run(".", "stage_02", use_conda = False)