import transformers
import argparse
import torch
import bitsandbytes as bnb
import argparse
from transformers import (
    
    AutoModel,AutoModelForCausalLM
)
def print_all_linear_names(model):
    for name, module in model.named_modules():
        print(" module name "+name+" type "+str(type(module)))
        if name.endswith("embed_tokens"):
            print("   num_embeddings  "+str(module.num_embeddings)+" embedding_dim "+str(module.embedding_dim)+ " size "+str(module.parameters))
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, bnb.nn.Linear8bitLt) or  isinstance(module, torch.nn.Linear):
            print("type "+str(type(module.weight.data)) +"   weight "+str(module.weight.data.shape) + " size "+str(module.weight.data.size)  )
def main(model_name:str):
    print("Hello, LLaMa World!")
    #model_name ="/models/base/llama-13b-hf"
    #model_name ="/models/base/llama-13b-hf"
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto")
    print(model)
    print_all_linear_names(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, required=True)
    args = parser.parse_args()
    main(args.base_model_path)

