from src.utils.all_utils import create_directory, read_yaml, copy_file, get_data, save_json
from src.utils.featurize import save_matrix
import argparse
import os
from pprint import pprint
import logging
import io
import numpy as np
import joblib
import sklearn.metrics as metrics
import json
import math

logging_str = "[%(asctime)s:  %(levelname)s: %(module)s]:  %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir, "running_logs.log"),
                                            level = logging.INFO,
                                            format = logging_str, 
                                            filemode = 'a')  



def evaluation(config_path, params_path):
    ## converting xml data to tsv
    config = read_yaml(config_path)
    params = read_yaml(params_path)
        
    artifacts = config['artifacts']
    
    featurized_data_dir_path = os.path.join(artifacts['Artifacts_dir'], artifacts['Featurized_Data'])
    featurized_test_data_path = os.path.join(featurized_data_dir_path, artifacts['Featurized_Out_Test'])
    
    model_dir_path = os.path.join(artifacts['Artifacts_dir'], artifacts['Model_dir'])
    model_path = os.path.join(model_dir_path, artifacts['Model_Name'])
    
    model = joblib.load(model_path)
    test_matrix = joblib.load(featurized_test_data_path)
    
    labels = np.squeeze(test_matrix[:, 1].toarray())
    X = test_matrix[:, 2:]
    
    prediction_by_class = model.predict_proba(X)
    predictions = prediction_by_class[ :, 1]
            
    PRC_json_path = config["plots"]["PRC"]
    ROC_json_path = config["plots"]["ROC"]
    
    scores_json_path = config["metrics"]["Scores"]
    
    avg_prec = metrics.average_precision_score(labels, predictions)
    roc_auc = metrics.roc_auc_score(labels, predictions)
    
    scores = {
        "avg_prec": avg_prec,
        "roc_auc": roc_auc
    }
    
    save_json(scores_json_path, scores)
    
    precision, recall, prc_threshold = metrics.precision_recall_curve(labels, predictions)
    
    nth_point = math.ceil(len(prc_threshold)/1000) 
    prc_points = list(zip(precision, recall, prc_threshold))[::nth_point]
    
    prc_data = {
        "prc": [
            {"precision": p, "recall": r, "threshold": t}
            for p, r, t in prc_points
        ]
    }
    
    save_json(PRC_json_path, prc_data)
    
    fpr, tpr, roc_threshold = metrics.roc_curve(labels, predictions)
    
    roc_data = {
        "roc": [
            {"fpr": fp, "tpr": tp, "threshold": t}
            for fp, tp, t in zip(fpr, tpr, roc_threshold)
        ]
    }
    
    save_json(ROC_json_path, roc_data)
    
            
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "config/config.yaml")
    args.add_argument("--params", "-p", default = "params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info(">>>>> Stage Four started")
        evaluation(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info("Stage Four completed!!! >>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e
     
    