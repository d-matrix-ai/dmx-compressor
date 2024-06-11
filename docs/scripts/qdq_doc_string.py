from mltools.utils.fx.visualize_graph import visualize_graph
import torch
from mltools.dmx import nn

BATCH_SIZE = 16
SEQ_LEN = 1024
HID_DIM = 64

model_dict = {
    "ResAdd": nn.ResAdd(),
    "ActActMatMul": nn.ActActMatMul(),
    "Linear": nn.Linear(HID_DIM, 32),
    "Embedding": nn.Embedding(5000, 100),
    "SoftMax": nn.Softmax(),
    "LayerNorm": nn.LayerNorm(HID_DIM),
    "Dropout": nn.Dropout(),
    "GELU": nn.GELU(),
}


for name, layer in model_dict.items():
    gm = torch.fx.GraphModule(layer, layer.to_compiler_graph())
    print(name)
    gm.graph.print_tabular()
    print("\n\n")
