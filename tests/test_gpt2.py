#!/usr/bin/env python3

import torch.fx as fx

import transformers
import torch
from mltools.fx.transformer import NodeDictTransformer
from torch.fx import GraphModule
from mltools.utils.fx.visualize_graph import GraphvizInterpreter
import transformers.utils.fx as hf_fx

model = transformers.AutoModel.from_pretrained('gpt2')
# prepare and test input
DEFAULT_INPUT_IDS = [[7454, 2402, 257, 640]]
x = torch.LongTensor(DEFAULT_INPUT_IDS)
gm = hf_fx.symbolic_trace(model)
node_dict = NodeDictTransformer(gm).transform()
gi = GraphvizInterpreter(gm, node_dict)
gi.run(x)
print("done")
gi.pygraph.render(filename = './doctest-output/'+'test_gpt2')
