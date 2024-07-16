import random
import torch
from datasets import Dataset
import warnings
from torch.utils.data import IterableDataset

# COPIED: https://github.com/huggingface/trl/blob/052a8e14b51161a5b7f7bc723ae2fa0144d76fce/trl/trainer/utils.py#L426C1-L546C1
class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        if formatting_func is not None:
            if formatting_func.__code__.co_argcount > 1:
                warnings.warn(
                    "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, add_special_tokens=self.add_special_tokens, truncation=False)[
                "input_ids"
            ]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }

# COPIED from: https://github.com/huggingface/trl/blob/052a8e14b51161a5b7f7bc723ae2fa0144d76fce/trl/trainer/sft_trainer.py#L597
def create_packed_dataset(dataset, tokenizer, max_seq_len=2048):
    formatting_func = lambda x: tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)  
    
    constant_length_iterator = ConstantLengthDataset(
        tokenizer,
        dataset,
        formatting_func=formatting_func,
        seq_length=max_seq_len,
        infinite=False,
        add_special_tokens=False,
    )
    
    def data_generator(constant_length_iterator):
      yield from constant_length_iterator
    
    packed_dataset = Dataset.from_generator(
            data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
        )
    
    return packed_dataset
