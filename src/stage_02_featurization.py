from src.utils.all_utils import create_directory, read_yaml, copy_file, get_data
from src.utils.featurize import save_matrix
import argparse
import os
from pprint import pprint
import logging
import io
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


logging_str = "[%(asctime)s:  %(levelname)s: %(module)s]:  %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir, "running_logs.log"),
                                            level = logging.INFO,
                                            format = logging_str, 
                                            filemode = 'a')  


def featurization(config_path, params_path):
    ## converting xml data to tsv
    config = read_yaml(config_path)
    params = read_yaml(params_path)
        
    artifacts = config['artifacts']
    prepared_data_dir_path = os.path.join(artifacts['Artifacts_dir'], artifacts['Prepared_Data_dir'])
    
    train_data_path = os.path.join(prepared_data_dir_path, artifacts['Trained_Data'])
    test_data_path = os.path.join(prepared_data_dir_path, artifacts['Test_Data'])
    
    featurized_data_dir_path = os.path.join(artifacts['Artifacts_dir'], artifacts['Featurized_Data'])
    create_directory([featurized_data_dir_path])
    
    featurized_train_data_path = os.path.join(featurized_data_dir_path, artifacts['Featurized_Out_Train'])
    featurized_test_data_path = os.path.join(featurized_data_dir_path, artifacts['Featurized_Out_Test'])
    
    max_features = params['featurize']['max_features']
    ngrams = params['featurize']['ngrams']
    
    ############## Get Train Data  ############################

    df_train = get_data(train_data_path)
    
    train_words = np.array(df_train.text.str.lower().astype('U').values)
    
    ## Bag Of Words
    bag_of_words = CountVectorizer(
        stop_words = "english", max_features = max_features, ngram_range = (1, ngrams)
    )

    bag_of_words.fit(train_words)
    train_words_binary_matrix = bag_of_words.transform(train_words)
    
    tfidf = TfidfTransformer(smooth_idf = False)
    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)
    save_matrix(df_train, train_words_tfidf_matrix, featurized_train_data_path)
    
    ############# Get Test Data  ####################
    
    df_test = get_data(test_data_path)
    test_words = np.array(df_test.text.str.lower().values.astype("U1000")) 
    test_words_binary_matrix = bag_of_words.transform(test_words)
    test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)
    
    save_matrix(df_test, test_words_tfidf_matrix, featurized_test_data_path)
    
    
    
            
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "config/config.yaml")
    args.add_argument("--params", "-p", default = "params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info(">>>>> Stage two started")
        featurization(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info("Stage two completed!!! >>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e
     
    