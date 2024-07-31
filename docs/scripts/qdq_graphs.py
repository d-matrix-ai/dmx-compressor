from dmx.compressor.utils.fx.visualize_graph import visualize_graph
import torch
from dmx.compressor.dmx import nn

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


def create_input(layer):
    if isinstance(layer, nn.Linear):
        return (torch.rand(BATCH_SIZE, SEQ_LEN, layer.in_features),)
    elif isinstance(layer, nn.Embedding):
        return (torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN)),)
    elif isinstance(layer, nn.LayerNorm):
        return (torch.rand(BATCH_SIZE, SEQ_LEN, HID_DIM),)
    elif isinstance(layer, nn.ResAdd):
        return (
            torch.rand(BATCH_SIZE, SEQ_LEN, HID_DIM),
            torch.rand(BATCH_SIZE, SEQ_LEN, HID_DIM),
        )
    elif isinstance(layer, nn.ActActMatMul):
        return (
            torch.rand(BATCH_SIZE, SEQ_LEN, HID_DIM),
            torch.rand(BATCH_SIZE, HID_DIM, SEQ_LEN),
        )
    else:
        return (torch.rand(BATCH_SIZE, SEQ_LEN, HID_DIM),)


for name, layer in model_dict.items():
    gm = torch.fx.GraphModule(layer, layer.to_compiler_graph())
    inp = create_input(layer)
    visualize_graph(gm, input=inp, file_name=name + "_qdq")
