import os
import sys
import copy
import logging
from typing import List,Sequence
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
import fire
import torch
import prompt
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer,LlamaTokenizerFast
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
IGNORE_INDEX = -100
def check_and_fix_special_tokens(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    logging.info(f" token class ,tokens vocab :{len(tokenizer.get_vocab())}\n {tokenizer}")
    default_pad = "[PAD]"
    default_pad=   default_pad if tokenizer.pad_token is None else tokenizer.pad_token     
    if len(tokenizer.get_vocab())==32000:
       added=tokenizer.add_special_tokens({'pad_token': default_pad})
       logging.info(f"add pad token because tokenizer size is {len(tokenizer.get_vocab())},added={added}") 
    logging.info(f'{tokenizer.bos_token=} ,bos_token_id={tokenizer.bos_token_id}')
    logging.info(f'{tokenizer.eos_token=} ,eos_token_id={tokenizer.eos_token_id}')
    logging.info(f'{tokenizer.pad_token=} ,pad_token_id={tokenizer.pad_token_id}')
    logging.info(f'{tokenizer.unk_token=} ,unk_token_id={tokenizer.unk_token_id}')
    logging.info(f'{tokenizer.all_special_tokens=}')
    #tokenizer.pad_token = tokenizer.unk_token
    
    if tokenizer.eos_token_id != model.config.eos_token_id :
         logging.info("Error tokenizer.eos_token_id not equals model ",tokenizer.eos_token_id,model.config.eos_token_id)
         exit(-1)
    if tokenizer.eos_token_id !=2 or   tokenizer.bos_token_id!=1:
          logging.info("Error tokenizer.bos_token_id or bos_token_id eror, should 1 and 2,cur ",tokenizer.eos_token_id,model.config.eos_token_id)
          exit(-1)           
            
    if tokenizer.pad_token_id != model.config.pad_token_id :
         logging.info("warn tokenizer.pad_token_id not equals model ",tokenizer.pad_token_id,model.config.pad_token_id," set model token to tokenizer token to fix it")
         model.config.pad_token_id =tokenizer.pad_token_id      


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self,template:str, tokenizer: transformers.PreTrainedTokenizer, source_max_len: int,target_max_len: int,train_on_input: bool,data_path:str,max_train_samples:int):
        super(SupervisedDataset, self).__init__()
        logging.info("Loading data...")
        # Load dataset.
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=data_path)
        else:
            dataset = load_dataset(self.data_path)
        
        train_dataset = dataset['train']
        if max_train_samples is not None and len(train_dataset) > max_train_samples:
                train_dataset = train_dataset.select(range(max_train_samples))
        logging.info(f'train data size {len(train_dataset)}')
        logging.info("Tokenizing inputs... This may take some time...")
        input_ids_all=[]
        lables_all=[]
        for example in train_dataset :
            source,target=prompt.get_templated_source_target(template,example,tokenizer) 
             
            # Tokenize
            tokenized_source = tokenizer.__call__(
                source,
                max_length=source_max_len,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt"
            )
            tokenized_target = tokenizer.__call__(
                target,
                max_length=target_max_len,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt"
            )
            # Build the input and labels for causal LM
            input_ids=torch.cat((tokenized_source.input_ids[0] , tokenized_target.input_ids[0]),dim=0)
            labels = copy.deepcopy(input_ids)
            if not train_on_input:
               labels[:len(tokenized_source.input_ids[0])] = IGNORE_INDEX
            
            input_ids_all.append(input_ids)
            lables_all.append(labels)
            #print(input_ids)
            #print("labels ",labels)            
        
        self.input_ids = input_ids_all
        self.labels =lables_all

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        rest=dict(input_ids=self.input_ids[i], labels=self.labels[i])
        #print(rest)
        return rest
       
  
class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logging.info('Saving PEFT checkpoint...')
        checkpoint_folder=""
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        logging.info("save path ",pytorch_model_path)


    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)




def train(
    # model/data params
    base_model: str = "",
    data_path: str = "yahma/alpaca-cleaned",
    template:str="ALPACA_PROMP",
    output_dir: str = "./lora-alpaca",
    device_map: str = "auto",
    # training hyperparams
    gradient_accumulation_steps:int=4,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    warmup_ratio: float =0.03,
    warmup_steps: int =100,
    source_max_len: int = 128,
    target_max_len: int = 1024,
    train_on_input:bool =False,
    max_train_samples:int=None,
    optim:str="paged_adamw_32bit",
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    gradient_checkpointing = True,
    lora_target_modules: List[str] = [
       "q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"
    ],
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve

    resume_from_checkpoint: str = None,  # either training train_on_sourcecheckpoint or final adapter
    bf16:bool =False

):
    major, minor = torch.cuda.get_device_capability()
    if torch.cuda.is_bf16_supported():
            logging.info('='*80)
            logging.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logging.info('='*80)
            if bf16 is None:
                bf16 =True
    

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
            
    #device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    
    logging.info(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"template: {template}\n"
            f"output_dir: {output_dir}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"gradient_accumulation_steps {gradient_accumulation_steps}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"source_max_len: {source_max_len}\n"
            f"target_max_len: {target_max_len}\n"
            f"max_train_samples: {max_train_samples}\n"
            f"gradient_checkpointing {gradient_checkpointing}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train on input part: {train_on_input}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint}\n"
            f"optim: {optim}\n"
           
        )

    
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map=device_map,
            trust_remote_code=True
        )
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        padding_side="right"

    )
    #print("origin model\n",model)
    check_and_fix_special_tokens(tokenizer,model)
    train_dataset=SupervisedDataset(template,tokenizer,source_max_len,target_max_len,train_on_input,data_path,max_train_samples)
    data_collator=DataCollatorForCausalLM(tokenizer)
    if gradient_checkpointing:
       model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    #model = prepare_model_for_kbit_training(model)
    logging.info(f"train model \n {model}")
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if resume_from_checkpoint:
            # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            logging.info(f"Restarting from {checkpoint_name}")
            model = PeftModel.from_pretrained(model, join(checkpoint_name, 'adapter_model'), is_trainable=True)
        else:
            logging.info(f"Checkpoint {checkpoint_name} not found,no load ")
            model = get_peft_model(model, config)
    else:
        model = get_peft_model(model, config)
    
    #print("peft train model \n",model)
    model.print_trainable_parameters()
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        #eval_dataset=data_module['eval_dataset'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=bf16,
            logging_steps=10,
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=100 ,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            gradient_checkpointing=gradient_checkpointing,
            report_to=None,
            optim=optim

        ),
        callbacks=[SavePeftModelCallback]
    )
    model.config.use_cache = False
    logging.info(f"DataLoad :{trainer.get_train_dataloader()}")
    
    trainer.train()

    model.save_pretrained(output_dir)

    logging.info(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
            


if __name__ == "__main__":
    #Creating and Configuring Logger
    logging.basicConfig(level=logging.INFO)
    fire.Fire(train)
