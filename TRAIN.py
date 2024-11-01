import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from filelock import FileLock
from torch.utils.data import random_split
import tqdm
from tqdm import tqdm
from typing import Dict

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint, DataConfig, ScalingConfig
from ray.train.torch import TorchTrainer

#import evaluate
import torch
from torch.optim import AdamW
from accelerate import Accelerator
import pandas as pd

import DataGenerator
from DataGenerator import Generator
from torch.utils.data import Dataset,DataLoader

from GPT import MODEL
torch.manual_seed(0)
from transformers import T5ForConditionalGeneration,T5Tokenizer
import sys
import warnings
warnings.filterwarnings(action='ignore')

import logging  
logging.getLogger().setLevel(logging.ERROR)  # Suppresses INFO and DEBUG messages

def TrainGpt(config):
    
    """Your training function that launches on each worker."""
    
    NUM_ENCODER_LAYER=config['NUM_ENCODER_LAYER']
    NUM_DECODER_LAYER=config['NUM_DECODER_LAYER']
    NUM_ENCODER_HEAD=config['NUM_ENCODER_HEAD']
    NUM_DECODER_HEAD=config['NUM_DECODER_HEAD']
    ENCODER_EMBED_DIM=config['ENCODER_EMBED_DIM']
    DECODER_EMBED_DIM=config['DECODER_EMBED_DIM']
    ENCODER_MLP_ratio=config['ENCODER_MLP_ratio']
    DECODER_MLP_ratio=config['DECODER_MLP_ratio']
    VOCAB_SIZE=config['VOCAB_SIZE']  
    lr=config['lr']
    EPOCHS=  config['EPOCHS']
    SEED=config['SEED']
    TRAIN_PATH=config['TRAIN_PATH']
    VALID_PATH=config['VALID_PATH']
    TOKENIZER=config['TOKENIZER']
    BATCH_SIZE=config['BATCH_SIZE']
    
    
    
  
    train_df=pd.read_csv(TRAIN_PATH)
    valid_df=pd.read_csv(VALID_PATH)
    
    train_data=Generator(dataset=train_df,tokenizer=TOKENIZER,padding_style=None)
    train_dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,persistent_workers=True)
    
    valid_data=Generator(dataset=valid_df,tokenizer=TOKENIZER,padding_style=None)
    valid_dataloader=DataLoader(valid_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,persistent_workers=True)
    
    accelerator = Accelerator(cpu=True)
    
    model=MODEL(
                NUM_ENCODER_LAYER,
                NUM_DECODER_LAYER,
                NUM_ENCODER_HEAD,
                NUM_DECODER_HEAD,
                ENCODER_EMBED_DIM,
                DECODER_EMBED_DIM,
                ENCODER_MLP_ratio,
                DECODER_MLP_ratio,
                VOCAB_SIZE
                )
    
    optimizer = AdamW(params=model.parameters(), lr=lr)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        for batch in tqdm(train_dataloader,dynamic_ncols=False,desc=f'Epoch {epoch}/{EPOCHS}',bar_format='{desc}|{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',leave=False,colour='green'):
            batch_data={'input_ids':batch['input'].to(accelerator.device),'attention_mask':batch['attention_mask'].to(accelerator.device),'label_ids':batch['label'].to(accelerator.device)}
            # print(batch_data['input_ids'].shape)
            # print(batch_data['label_ids'].shape)
            outputs = model(**batch_data)
            loss = outputs['loss']
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
if __name__=="__main__":
    MODEL_NAME='t5-base'
    model=T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer=T5Tokenizer.from_pretrained(MODEL_NAME)

    config={
        'NUM_ENCODER_LAYER':2,
        'NUM_DECODER_LAYER':2,
        'NUM_ENCODER_HEAD': 2,
        'NUM_DECODER_HEAD': 2,
        'ENCODER_EMBED_DIM': 64,
        'DECODER_EMBED_DIM': 64,
        'ENCODER_MLP_ratio': 2,
        'DECODER_MLP_ratio': 2,
        'VOCAB_SIZE':len(tokenizer),
        'lr':0.0001,
        'EPOCHS':2,
        'SEED':0,
        'TRAIN_PATH':'/content/MINI-GPT/DATA/Train.csv',
        'VALID_PATH':'/content/MINI-GPT/DATA/Test.csv',
        'TOKENIZER':tokenizer,
        'BATCH_SIZE':16
            }
    # df=pd.read_csv(config['TRAIN_PATH'])
    # print(df.head(3))
    # sys.exit()
    resources={"CPU": 7}
    ray.init()
    trainer = TorchTrainer(
                            TrainGpt,
                            train_loop_config=config,
                            scaling_config=ScalingConfig(num_workers=1, 
                                                         use_gpu=True,
                                                         #trainer_resources=resources,
                                                         #resources_per_worker={'CPU':}
                                                        ),
                            # run_config=ray.train.RunConfig(failure_config=train.FailureConfig(-1))
                        )

    result = trainer.fit()

    