from .visualization import mask2braille, print_model_tree
from .io import load_config_file, save_config_file
from . import dmir_pb2
from .graph_utils import graph_dict_to_csv, save_to_file, prune_graph

def run_protoc_dmir():
    import os
    dmir_proto_dir = os.path.dirname(__file__)
    dmir_proto_file = "dmir.proto"
    if os.path.exists(os.path.join(dmir_proto_dir, dmir_proto_file)):
        os.system(
            f"protoc -I={dmir_proto_dir} --python_out={dmir_proto_dir} {os.path.join(dmir_proto_dir, dmir_proto_file)}"
        )
