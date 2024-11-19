import sys
import os
from dataclasses import dataclass, field
from peft import LoraConfig

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from trl import TrlParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer
from optimum.neuron.distributed import lazy_load_for_parallelism

LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

def training_function(script_args, training_args):
    # load dataset and tokenizer
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    # load model from the hub with a bnb config
    with lazy_load_for_parallelism(tensor_parallel_size=training_args.tensor_parallel_size):
        model = AutoModelForCausalLM.from_pretrained(
                script_args.model_id,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                use_cache=False if training_args.gradient_checkpointing else True,
        )
        
    # Create LoRA configuration
    # config = LoraConfig(
    #     lora_alpha=8,
    #     lora_dropout=0.05,
    #     r=16,
    #     bias="none",
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    # )
    
    

    # Create Trainer instance
    trainer = NeuronSFTTrainer(
        model=model,
        args=training_args,
        # peft_config=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )

    # Start training
    trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload

@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: str = field(
        metadata={"help": "Path to the dataset with conversational or instruction format."},
        default=None,
    )


def main():
    parser = TrlParser([ScriptArguments, NeuronSFTConfig])
    script_args, training_args = parser.parse_args_and_config()    

    # set seed
    set_seed(training_args.seed)

    # run training function
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()
