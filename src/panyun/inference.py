import argparse
import json, os
import torch
import sys
from os.path import join
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
from peft import  PeftModel

generation_config = dict(
    temperature=0.001,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
    )


sample_data = ["为什么要减少污染，保护环境？"]


def inference(args):
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )
    tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            padding_side='right',
            legacy=False,
            use_fast=True,
        )
    if tokenizer.pad_token is None:
        print("model no padding token ,set to eos pad \r\n")
        tokenizer.pad_token=tokenizer.eos_token
        tokenizer.pad_token_id=tokenizer.eos_token_id
        
    if args.lora_path is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, join(args.lora_path, 'adapter_model'),torch_dtype=load_type,device_map='auto',)
    else:
        model = base_model



    
    
    if device==torch.device('cpu'):
        model.float()
    
    model.eval()
    # test data
    if args.data_file is not None:
        with open(args.data_file,'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)


    with torch.no_grad():
        if args.data_file is  None:
            print("Start inference with instruction mode.")

            print('='*85)
            print("+ 该模式下仅支持单轮问答，无多轮对话能力。ctrl+d 结束一次输入\n")
            print("+ This mode only supports single-turn QA.\n")
            print('='*85)

            while True:
                print("input:\r\n")
                lines=sys.stdin.readlines()
                input_text=""
                for line in lines:
                    input_text+=line
                
                if len(input_text.strip())<=1:
                    break

                print("waiting for answer ....\r\n")
                inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids = inputs["input_ids"].to(device),
                    attention_mask = inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s,skip_special_tokens=True)
                response = output
                print("Response: ",response)
                print("\n")
        else:
            print("Start inference.")
            results = []
            for index, example in enumerate(examples):
                input_text = example
                inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids = inputs["input_ids"].to(device),
                    attention_mask = inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s,skip_special_tokens=True)
                response = output
                print(f"======={index}=======")
                print(f"Input: {example}\n")
                print(f"Output: {response}\n")

                results.append({"Input":input_text,"Output":response})

            dirname = os.path.dirname(args.predictions_file)
            os.makedirs(dirname,exist_ok=True)
            with open(args.predictions_file,'w') as f:
                json.dump(results,f,ensure_ascii=False,indent=2)
            with open(dirname+'/generation_config.json','w') as f:
                json.dump(generation_config,f,ensure_ascii=False,indent=2)


if __name__ == '__main__':
    seed=1000
     # Set the seeds for reproducibility
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_path', default=None, type=str,help="If None, perform inference on the base model")
    parser.add_argument('--data_file',default=None, type=str,help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--interactive',default=None,action='store_true',help="run in the instruction mode (single-turn)")
    parser.add_argument('--predictions_file', default='./predictions.json', type=str)
    parser.add_argument('--load_in_8bit',default=True,action='store_true', help="Load the LLM in the 8bit mode")
    args = parser.parse_args()
    inference(args)