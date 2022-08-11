from nntplib import ArticleInfo
from src.utils.all_utils import  create_directory, read_yaml
from src.utils.model_utils import save_binary
import argparse
import os
from pprint import pprint
import logging
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch 


STAGE = "GET_DATA"

logging_str = "[%(asctime)s:  %(levelname)s: %(module)s]:  %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir, "running_logs.log"),
                                            level = logging.INFO,
                                            format = logging_str, 
                                            filemode = 'a')   
                         
        
def get_data(config_path):
    ## converting xml data to tsv
    config = read_yaml(config_path)
    
    logging.info("define Training kwargs")
    train_kwargs = {"batch_size" : config["params"]["BATCH_SIZE"]}
    test_kwargs = {"batch_size" : config["params"]["TEST_BATCH_SIZE"]}
    
    logging.info("Updating device configurationas per cuda availability")
    device_config = {"DEVICE" : 'cuda' if torch.cuda.is_available() else 'cpu'}
    config.update(device_config)
    
    
    if config['DEVICE'] == "cuda":
        cuda_kwargs = {"num_workers" : 1, "pin_memory" : True, "shuffle" : True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    
    logging.info("transforms added")
    transform = transforms.Compose(
            [transforms.ToTensor()]
    )
    
    logging.info("downloading training and testing data")
    train = datasets.MNIST(config["source_data_dirs"]["data"], train = True, download = True, transform = transform)
    test = datasets.MNIST(config["source_data_dirs"]["data"], train = False, transform = transform)

    logging.info("defining training and testing data loader")
    train_loader = torch.utils.data.DataLoader(train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test, **test_kwargs)
    
    artifacts = config['artifacts']
    model_config_filepath = os.path.join(artifacts['artifacts'], artifacts["model_config_dir"])
    create_directory([model_config_filepath])
    train_loader_bin_file = artifacts["train_loader_bin"]
    test_loader_bin_file = artifacts["test_loader_bin"]
    
    train_loader_bin_filepath = os.path.join(model_config_filepath, train_loader_bin_file)
    test_loader_bin_filepath = os.path.join(model_config_filepath, test_loader_bin_file)
    
    save_binary(train_loader, train_loader_bin_filepath)
    save_binary(test_loader, test_loader_bin_filepath)
                

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "configs/config.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info(">>>>> Stage one started")
        get_data(config_path = parsed_args.config)
        logging.info("Stage one completed!!! >>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e
     
    