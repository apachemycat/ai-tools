IGNORE_INDEX = -100
ALL_PROMPT_TEMPLATES= {
    'ALPACA_PROMP' : {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
      },
    'PANYUN_PROMP':{
        "prompt_input": (
            "您是一个磐云专家，写出以下问题的解答\n\n"
            " 指令:\n{instruction}\n\n参数:\n{input}\n\n 应答: "
        ),
        "prompt_no_input": (
            "您是一个磐云专家，写出以下问题的解答\n\n"
            " 指令:\n{instruction}\n\n 应答: "
        ),
    }
    }

def get_templated_source_target(template:str,example:str,tokenizer):
    #print("get template ",template)
    templateDic=ALL_PROMPT_TEMPLATES[template]
    if example.get("input", "") != "":
        prompt_format =  templateDic["prompt_input"]
    else:
        prompt_format = templateDic["prompt_no_input"]
    input= prompt_format.format(**example)
    source = f"{tokenizer.bos_token}{input}" 
    #print(source)
    target = f"{example['output']}{tokenizer.eos_token}" 
    #print(target)
    return  source, target
    
