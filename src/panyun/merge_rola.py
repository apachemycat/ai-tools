"""
Apply the LoRA weights on top of a base model.

"""
import argparse
from typing import Tuple
from os.path import join
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM,AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


def apply_lora(
    base_model_path: str,
    lora_path: str,
    load_8bit: bool = False,
    target_model_path: str = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Applies the LoRA adapter to a base model and saves the resulting target model (optional).

    Args:
        base_model_path (str): The path to the base model to which the LoRA adapter will be applied.
        lora_path (str): The path to the LoRA adapter.
        target_model_path (str): The path where the target model will be saved (if `save_target_model=True`).


    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the target model and its tokenizer.

    """
    # Load the base model and tokenizer
    print(f'Loading the base model from {base_model_path}')
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=load_8bit,
        device_map='auto',
    )



    # Load the LoRA adapter
    print(f'Loading the LoRA adapter from {lora_path}')
    try:
      lora_model = PeftModel.from_pretrained(base_model, join(lora_path, 'adapter_model'))
    except Exception as e:
        print(e)  
    print("merge Lora apater ...")
    base_model  = lora_model.merge_and_unload()
    print("save merged model...")

    print(f'Saving the target model to {target_model_path}')
    LlamaForCausalLM.save_pretrained(base_model, target_model_path)


    # Load the tokenizer
    if base_model.config.model_type == 'llama':
        # Due to the name of Transformers' LlamaTokenizer, we have to do this
        base_tokenizer = LlamaTokenizer.from_pretrained(
            base_model_path,
            padding_side='right',
            legacy=False,
            use_fast=True,
        )
    else:
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            padding_side='right',
            legacy=False,
            use_fast=True,
        )
    base_tokenizer.save_pretrained(target_model_path)
    return lora_model, base_tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--target_model_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--load_8bit', type=bool, required=False)

    

    args = parser.parse_args()

    apply_lora(args.base_model_path,args.lora_path, args.load_8bit,args.target_model_path)
