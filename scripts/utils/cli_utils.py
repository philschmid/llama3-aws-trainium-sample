import os 
import sys
# from optimum.neuron import NeuronHfArgumentParser
from transformers import HfArgumentParser as NeuronHfArgumentParser
import yaml
from argparse import Namespace

class YamlConfigParser:
    def parse_and_set_env(self, config_path):
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)

        if "env" in config:
            env_vars = config.pop("env")
            if isinstance(env_vars, dict):
                for key, value in env_vars.items():
                    os.environ[key] = str(value)
            else:
                raise ValueError("`env` field should be a dict in the YAML file.")

        return config

    def to_string(self, config):
        final_string = """"""
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                if len(value) != 0:
                    value = str(value)
                    value = value.replace("'", '"')
                    value = f"'{value}'"
                else:
                    continue

            final_string += f"--{key} {value} "
        return final_string
      

class TrlParser(NeuronHfArgumentParser):
    def __init__(self, parsers, ignore_extra_args=False):
        """
        The TRL parser parses a list of parsers (TrainingArguments, trl.ModelConfig, etc.), creates a config
        parsers for users that pass a valid `config` field and merge the values that are set in the config
        with the processed parsers.

        Args:
            parsers (`List[argparse.ArgumentParser`]):
                List of parsers.
            ignore_extra_args (`bool`):
                Whether to ignore extra arguments passed by the config
                and not raise errors.
        """
        super().__init__(parsers)
        self.yaml_parser = YamlConfigParser()
        self.ignore_extra_args = ignore_extra_args

    def post_process_dataclasses(self, dataclasses):
        # Apply additional post-processing in case some arguments needs a special
        # care
        training_args = trl_args = None
        training_args_index = None

        for i, dataclass_obj in enumerate(dataclasses):
            if dataclass_obj.__class__.__name__ == "TrainingArguments":
                training_args = dataclass_obj
                training_args_index = i
            elif dataclass_obj.__class__.__name__ in ("SFTScriptArguments", "DPOScriptArguments"):
                trl_args = dataclass_obj
            else:
                ...

        if trl_args is not None and training_args is not None:
            training_args.gradient_checkpointing_kwargs = dict(
                use_reentrant=trl_args.gradient_checkpointing_use_reentrant
            )
            dataclasses[training_args_index] = training_args

        return dataclasses

    def parse_args_and_config(self, return_remaining_strings=False):
        yaml_config = None
        if "--config" in sys.argv:
            config_index = sys.argv.index("--config")

            _ = sys.argv.pop(config_index)  # --config
            config_path = sys.argv.pop(config_index)  # path to config
            yaml_config = self.yaml_parser.parse_and_set_env(config_path)

            self.set_defaults_with_config(**yaml_config)

        outputs = self.parse_args_into_dataclasses(return_remaining_strings=return_remaining_strings)

        if yaml_config is None:
            return outputs

        if return_remaining_strings:
            # if we have extra yaml config and command line strings
            # outputs[-1] is remaining command line strings
            # outputs[-2] is remaining yaml config as Namespace
            # combine them into remaining strings object
            remaining_strings = outputs[-1] + [f"{key}: {value}" for key, value in vars(outputs[-2]).items()]
            return outputs[:-2], remaining_strings
        else:
            # outputs[-1] is either remaining yaml config as Namespace or parsed config as Dataclass
            if isinstance(outputs[-1], Namespace) and not self.ignore_extra_args:
                remaining_args = vars(outputs[-1])
                raise ValueError(f"Some specified config arguments are not used by the TrlParser: {remaining_args}")

            return outputs

    def set_defaults_with_config(self, **kwargs):
        """Defaults we're setting with config allow us to change to required = False"""
        self._defaults.update(kwargs)

        # if these defaults match any existing arguments, replace
        # the previous default on the object with the new one
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs[action.dest]
                action.required = False