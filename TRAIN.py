

import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
#import torch.nn.functional as F
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
from ray.experimental.tqdm_ray import tqdm

from tempfile import TemporaryDirectory

#import evaluate
import torch
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import pandas as pd

import DataGenerator
from DataGenerator import Generator
from torch.utils.data import Dataset,DataLoader

from GPT import MODEL
torch.manual_seed(0)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    T5ForConditionalGeneration,T5Tokenizer
)
import sys
import warnings
warnings.filterwarnings(action='ignore')

import logging  
logging.getLogger().setLevel(logging.ERROR)  # Suppresses INFO and DEBUG messages
import gc

import evaluate
import argparse

class ParseDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):

        d = getattr(namespace, self.dest) or {}

        if values:
            if type(values) is not list:
                values=[values]
            for item in values:
                split_items = item.split("=")
                key = split_items[
                    0
                ].strip()  # we remove blanks around keys, as is logical
                # print(split_items)
                value = split_items[1]

                d[key] = value

        setattr(namespace, self.dest, d)

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
    
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
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
    metric = evaluate.load("accuracy")
    optimizer = AdamW(params=model.parameters(), lr=lr)
    
    steps_per_epoch = len(train_data) // (accelerator.num_processes * BATCH_SIZE)   
    
    lr_scheduler = get_linear_schedule_with_warmup(
                                                    optimizer=optimizer,
                                                    num_warmup_steps=100,
                                                    num_training_steps=(steps_per_epoch * EPOCHS),
                                                    )
    model, optimizer, lr_scheduler= accelerator.prepare(model, optimizer,lr_scheduler)
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        # Use custom progress bar
        for batch in tqdm(train_dataloader,desc=f'Training| Epoch {epoch}/{EPOCHS}'):
            batch_data={'input_ids':batch['input'].to(accelerator.device),
                        'input_padding_mask':batch['input_attention_mask'].to(accelerator.device),
                        'target_padding_mask':batch['label_attention_mask'].to(accelerator.device),
                        'label_ids':batch['label'].to(accelerator.device)}
            # print(batch_data['input_padding_mask'].shape)
            # print(batch_data['target_padding_mask'].shape)
            # print(batch_data['label_ids'].shape)
            
            outputs = model(**batch_data)
            loss = outputs['loss']
            # print('loss:',loss)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        del batch,batch_data,outputs
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        model.eval()
        for batch in tqdm(valid_dataloader,desc=f'Validating| Epoch {epoch}/{EPOCHS}'):
            batch_data={'input_ids':batch['input'].to(accelerator.device),
                        'input_padding_mask':batch['input_attention_mask'].to(accelerator.device),
                        'target_padding_mask':batch['label_attention_mask'].to(accelerator.device),
                        'label_ids':batch['label'].to(accelerator.device)}
            with torch.no_grad():
                outputs = model(**batch_data,return_loss=False)
            predictions = outputs.argmax(dim=-1)

            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch_data["label_ids"])
            )
            
            predictions = predictions.view(-1)
            references = references.view(-1)
    
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            
            # del batch,batch_data,outputs
        
        eval_metric = metric.compute()
        accelerator.print(f"epoch {epoch}:", eval_metric) 
        # Report checkpoint and metrics to Ray Train
        # ==========================================
        with TemporaryDirectory() as tmpdir:
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model, f"{tmpdir}/ckpt_{epoch}.bin")
                checkpoint = Checkpoint.from_directory(tmpdir)
            else:
                checkpoint = None
            ray.train.report(metrics=eval_metric, checkpoint=checkpoint)
            
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu",type=bool)
    parser.add_argument('--resources_per_worker',action=ParseDict,metavar="KEY=VALUE",nargs="+")
    # print('hello')
    parser.add_argument('--trainer_resources',action=ParseDict,metavar="KEY=VALUE",nargs="+")
    
    parser.add_argument('--num_workers',type=int,nargs="+")
    args = parser.parse_args()
    
    use_gpu=args.use_gpu
    resources_per_worker=args.resources_per_worker
    resources_per_worker={k:int(v) for k,v in zip(list(resources_per_worker.keys()),list(resources_per_worker.values()))}
    
    trainer_resources=args.trainer_resources
    trainer_resources={k:int(v) for k,v in zip(list(trainer_resources.keys()),list(trainer_resources.values()))}
    # print(type(args.num_workers[0]))
    num_workers=args.num_workers[0]
    
    
    
    
    
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
        'EPOCHS':10,
        'SEED':0,
        'TRAIN_PATH':os.path.join(os.getcwd(),'DATA/Train.csv'),
        'VALID_PATH':os.path.join(os.getcwd(),'DATA/Train.csv'),
        'TOKENIZER':tokenizer,
        'BATCH_SIZE':4
            }
    # df=pd.read_csv(config['TRAIN_PATH'])
    # print(df.head(3))
    # sys.exit()
 
    # use_gpu=False
    # trainer_resources={"CPU":torch.get_num_threads()-1}
    # resources_per_worker={'CPU':1}
    # cuda_availability=torch.cuda.is_available()
    # if cuda_availability:
    #     num_workers=torch.cuda.device_count()
    #     use_gpu=True
    #     trainer_resources={'GPU':num_workers-1}
    #     resources_per_worker={'GPU':1}
        
        

    ray.init()
    trainer = TorchTrainer(
                            TrainGpt,
                            train_loop_config=config,
                            scaling_config=ScalingConfig(num_workers=num_workers, 
                                                         use_gpu=use_gpu,
                                                         trainer_resources=trainer_resources,
                                                         resources_per_worker=resources_per_worker
                                                        ),
                            # run_config=ray.train.RunConfig(failure_config=train.FailureConfig(-1))
                        )

    result = trainer.fit()

    
