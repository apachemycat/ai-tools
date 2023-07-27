import os
import sys
import copy
import logging
from typing import List,Sequence
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
import torch
import numpy as np
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer,LlamaTokenizerFast
import prompt
import argparse

if __name__ == "__main__":
 
        parser = argparse.ArgumentParser(description='Analyse train data set ')
        parser.add_argument('--data_path', required=True,
                            help='train data file ')
        parser.add_argument('--base_model', required=True,
                            help='base model path/name ')        
        parser.add_argument('--template',  required=True,
                            help=' prompt templates \n'+str(prompt.ALL_PROMPT_TEMPLATES.keys()))          
        args = parser.parse_args()
        data_path=args.data_path
        base_model=args.base_model
        template=args.template
        print(
                    f"Analyse instrcut train data :\n"
                    f"base_model: {base_model}\n"
                    f"data_path: {data_path}\n"
                    f"template: {template}\n"
                )
        logging.info("Loading data from ",data_path)
        # Load dataset.
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=data_path)
        else:
            dataset = load_dataset(data_path)
        
        train_dataset = dataset['train']
        print("train data size ",len(train_dataset))
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
          base_model,
           padding_side="right"
          )
      
        logging.info("Tokenizing inputs... This may take some time...")
        source_len_all=[]
        target_len_all=[]
        i =0
        for example in train_dataset :
             
            source,target=prompt.get_templated_source_target(template,example,tokenizer)    
            # Tokenize
            tokenized_source = tokenizer.__call__(
                source,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
            tokenized_target = tokenizer.__call__(
                target,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
            source_len=len(tokenized_source.input_ids[0])
            target_len=len(tokenized_target.input_ids[0])
            source_len_all.append(source_len)
            target_len_all.append(target_len)
            i+=1
            if i %100==0:
                print("finished ",i)
                
        print("source max len,",np.max(source_len_all)," median ",np.median(source_len_all)," min ",np.min(source_len_all)) 
        print("target max len,",np.max(target_len_all)," median ",np.median(target_len_all)," min ",np.min(target_len_all)) 

        
        
