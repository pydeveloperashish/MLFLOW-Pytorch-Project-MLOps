from src.utils.all_utils import create_directory, read_yaml, copy_file, get_data
from src.utils.featurize import save_matrix
import argparse
import os
from pprint import pprint
import logging
import io
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logging_str = "[%(asctime)s:  %(levelname)s: %(module)s]:  %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir, "running_logs.log"),
                                            level = logging.INFO,
                                            format = logging_str, 
                                            filemode = 'a')  


def train(config_path, params_path):
    ## converting xml data to tsv
    config = read_yaml(config_path)
    params = read_yaml(params_path)
        
    artifacts = config['artifacts']
    prepared_data_dir_path = os.path.join(artifacts['Artifacts_dir'], artifacts['Prepared_Data_dir'])
    
    featurized_data_dir_path = os.path.join(artifacts['Artifacts_dir'], artifacts['Featurized_Data'])
    featurized_train_data_path = os.path.join(featurized_data_dir_path, artifacts['Featurized_Out_Train'])
    
    model_dir_path = os.path.join(artifacts['Artifacts_dir'], artifacts['Model_dir'])
    create_directory([model_dir_path])
    model_path = os.path.join(model_dir_path, artifacts['Model_Name'])
    
    matrix = joblib.load(featurized_train_data_path)
    
    labels = np.squeeze(matrix[:, 1].toarray())
    X = matrix[:, 2:]
    
    logging.info(f"Input matrix size: {matrix.shape}")
    logging.info(f"X matrix size: {X.shape}")
    logging.info(f"Y matrix size or labels size: {labels.shape}")
    
    seed = params["train"]["seed"]
    n_est = params["train"]["n_est"]
    min_split = params["train"]["min_split"]
    
    model = RandomForestClassifier(
        n_estimators = n_est,
        min_samples_split = min_split,
        n_jobs = 2,
        random_state = seed    
    )
         
    model.fit(X, labels)
    joblib.dump(model, model_path)     
            
            
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "config/config.yaml")
    args.add_argument("--params", "-p", default = "params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info(">>>>> Stage three started")
        train(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info("Stage three completed!!! >>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e
     
    