from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def load_config_file(config_file="configs/corsair.yaml"):
    with open(config_file, "r") as f:
        config = load(f, Loader=Loader)
    return config

def save_config_file(config, config_file="configs/corsair.yaml"):
    with open(config_file, "w") as f:
        f.write(dump(config))