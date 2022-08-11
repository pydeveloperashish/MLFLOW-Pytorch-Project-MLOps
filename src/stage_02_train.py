from nntplib import ArticleInfo
from src.utils.all_utils import create_directory, read_yaml, copy_file, get_data
from src.utils.model_utils import ConvNet, load_binary
import argparse
import os
from pprint import pprint
import logging
import io
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch 

STAGE = "TRAIN MODEL"

logging_str = "[%(asctime)s:  %(levelname)s: %(module)s]:  %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir, "running_logs.log"),
                                            level = logging.INFO,
                                            format = logging_str, 
                                            filemode = 'a')  



def train_(config, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model.forward(data)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % config['params']['LOG_INTERVAL'] == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            if config['params']['DRY_RUN']:
                break 
            
            
def train(config_path):
    ## converting xml data to tsv
    config = read_yaml(config_path)
    
    logging.info("Updating device configurationas per cuda availability")
    device_config = {"DEVICE" : 'cuda' if torch.cuda.is_available() else 'cpu'}
    config.update(device_config)
    
    model = ConvNet().to(config['DEVICE'])
    scripted_model = torch.jit.script(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['params']['LR'])
    scheduler = StepLR(optimizer, step_size=1, gamma=config['params']['GAMMA'])
    
    
    
    
    artifacts = config['artifacts']
    model_config_filepath = os.path.join(artifacts['artifacts'], artifacts["model_config_dir"])
    train_loader_bin_file = artifacts["train_loader_bin"]
    train_loader_bin_filepath = os.path.join(model_config_filepath, train_loader_bin_file)
    train_loader = load_binary(train_loader_bin_filepath)
    
    
    # training loop
    for epoch in range(1, config['params']['EPOCHS'] + 1):
        train_(config, scripted_model, config['DEVICE'], train_loader, optimizer, epoch)
        scheduler.step()
        
    
            
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "configs/config.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info(">>>>> Stage two started")
        train(config_path = parsed_args.config)
        logging.info("Stage two completed!!! >>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e
     
    