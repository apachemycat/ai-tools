import transformers
import argparse
from transformers import (

    AutoModel,
)
def main():
    print("Hello, World!")
    #model_name ="/models/base/llama-13b-hf"
    model_name ="/models/WizardLM-13B-V1.0-Merged"
    model  = AutoModel.from_pretrained(
        model_name,
        device_map="auto")
    print(model)


if __name__ == "__main__":
    main()

