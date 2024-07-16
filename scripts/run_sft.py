import sys
import os
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from utils.cli_utils import TrlParser
from utils.dataset_utils import create_packed_dataset
from optimum.neuron import NeuronTrainer
from optimum.neuron import NeuronTrainingArguments
from optimum.neuron.distributed import lazy_load_for_parallelism

def training_function(script_args, training_args):
    # load dataset and tokenizer
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)

    # prepare and pack dataset
    dataset = create_packed_dataset(dataset, tokenizer, max_seq_len=script_args.max_seq_len)

    # load model from the hub with a bnb config
    with lazy_load_for_parallelism(tensor_parallel_size=training_args.tensor_parallel_size):
        model = AutoModelForCausalLM.from_pretrained(
                script_args.model_id,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                use_cache=False if training_args.gradient_checkpointing else True,
        )

    # Create Trainer instance
    trainer = NeuronTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,  # no special collator needed since we stacked the dataset
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
    max_seq_len: int = field(
        metadata={"help": "Maximum sequence length for the model."},
        default=1024,
    )


def main():
    parser = TrlParser([ScriptArguments, NeuronTrainingArguments])
    script_args, training_args = parser.parse_args_and_config()    

    # set seed
    set_seed(training_args.seed)

    # run training function
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()
