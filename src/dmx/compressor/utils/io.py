import torch
import os
from pathlib import Path
import hashlib
import tempfile
import shutil
import yaml
from typing import Type

from dmx.compressor.modeling.nn import DmxModule, DmxModuleType
from dmx.compressor.numerical import Format
from dmx.compressor.sparse import Sparseness
from dmx.compressor.functional import ApproximationFunction


def compute_md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_state_dict_and_return_url(module: torch.nn.Module, parent_dir: str) -> str:
    temp_file_name = tempfile.NamedTemporaryFile().name
    torch.save(module.state_dict(), temp_file_name)
    md5 = compute_md5(temp_file_name)
    file_name = os.path.join(parent_dir, f"{md5}.pth")
    shutil.move(temp_file_name, file_name)
    return Path(os.path.abspath(file_name)).as_uri()


def dmx_module_instance_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode
) -> str:
    return eval(f"nn.{node.value}")


def dmx_module_instance_representer(
    dumper: yaml.SafeDumper, val: Type[DmxModule]
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("!DmxModule", val.__name__, style=None)


def format_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> str:
    return Format.from_shorthand(node.value)


def format_representer(dumper: yaml.SafeDumper, val: Format) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("!Format", repr(val), style=None)


def sparseness_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> str:
    return Sparseness.from_shorthand(node.value)


def sparseness_representer(
    dumper: yaml.SafeDumper, val: Format
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("!Sparseness", repr(val), style=None)


def approximation_function_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode
) -> str:
    return ApproximationFunction.from_shorthand(node.value)


def approximation_function_representer(
    dumper: yaml.SafeDumper, val: Format
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("!ApproximationFunction", repr(val), style=None)


def get_loader():
    loader = yaml.SafeLoader
    loader.add_constructor("!DmxModule", dmx_module_instance_constructor)
    loader.add_constructor("!Format", format_constructor)
    loader.add_constructor("!Sparseness", sparseness_constructor)
    loader.add_constructor("!ApproximationFunction", approximation_function_constructor)
    return loader


def get_dumper():
    dumper = yaml.SafeDumper
    dumper.add_multi_representer(DmxModuleType, dmx_module_instance_representer)
    dumper.add_multi_representer(Format, format_representer)
    dumper.add_multi_representer(Sparseness, sparseness_representer)
    dumper.add_multi_representer(
        ApproximationFunction, approximation_function_representer
    )
    dumper.ignore_aliases = lambda _self, _data: True
    return dumper


def load_config_file(config_file="configs/dmx.yaml"):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=get_loader())
    return config


def save_config_file(config, config_file="configs/dmx.yaml"):
    with open(config_file, "w") as f:
        f.write(
            yaml.dump(
                config,
                Dumper=get_dumper(),
            )
        )
