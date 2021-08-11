# import torch
# import corsair

# from models.lenet import LeNet

# model = LeNet([512, 512])
# model.transform("./corsair_config.yaml")

# dump_dmir0(model, "./mnist-lenet-512-512.dmir0")

import utils.dmir_pb2 as dmir

model = dmir.Graph()
model.name = "lenet-512-512"

# model
_inp = model.input.add()
_inp.name = "images"
_inp.shape.append(1)
_inp.shape.append(784)
_inp.format = dmir.FLOAT

_out = model.output.add()
_out.name = "logits"
_out.shape.append(1)
_out.shape.append(10)
_out.format = dmir.FLOAT

_v0 = model.intermediate.add()
_v0.name = "v0"
_v0.shape.append(1)
_v0.shape.append(512)
_v0.format = dmir.FLOAT

_v1 = model.intermediate.add()
_v1.name = "v1"
_v1.shape.append(1)
_v1.shape.append(512)
_v1.format = dmir.FLOAT

_v2 = model.intermediate.add()
_v2.name = "v2"
_v2.shape.append(1)
_v2.shape.append(512)
_v2.format = dmir.FLOAT

_v3 = model.intermediate.add()
_v3.name = "v3"
_v3.shape.append(1)
_v3.shape.append(512)
_v3.format = dmir.FLOAT

layer1 = model.subgraph.add()
layer1.name = "layer1"
act_fn1 = model.subgraph.add()
act_fn1.name = "act_fn1"
act_fn1.instance = dmir.RELU
layer2 = model.subgraph.add()
layer2.name = "layer2"
act_fn2 = model.subgraph.add()
act_fn2.name = "act_fn2"
act_fn2.instance = dmir.RELU
layer3 = model.subgraph.add()
layer3.name = "layer3"

_1 = model.dependency.add()
_1.operation = layer1.name
_1.argument.append(_inp.name)
_1.result.append(_v0.name)

_2 = model.dependency.add()
_2.operation = act_fn1.name
_2.argument.append(_v0.name)
_2.result.append(_v1.name)

_3 = model.dependency.add()
_3.operation = layer2.name
_3.argument.append(_v1.name)
_3.result.append(_v2.name)

_4 = model.dependency.add()
_4.operation = act_fn2.name
_4.argument.append(_v2.name)
_4.result.append(_v3.name)

_5 = model.dependency.add()
_5.operation = layer3.name
_5.argument.append(_v3.name)
_5.result.append(_out.name)

# layer1
_inp = layer1.input.add()
_inp.name = "input"
_inp.shape.append(1)
_inp.shape.append(784)
_inp.format = dmir.FLOAT

_wt = layer1.input.add()
_wt.name = "weight"
_wt.shape.append(512)
_wt.shape.append(784)
_wt.format = dmir.FLOAT

_bs = layer1.input.add()
_bs.name = "bias"
_bs.shape.append(512)
_bs.format = dmir.FLOAT

_out = layer1.output.add()
_out.name = "output"
_out.shape.append(1)
_out.shape.append(512)
_out.format = dmir.FLOAT

_accum = layer1.intermediate.add()
_accum.name = "product"
_accum.shape.append(1)
_accum.shape.append(512)
_accum.format = dmir.FLOAT

_matmul = layer1.subgraph.add()
_matmul.name = "matmul"
_matmul.instance = dmir.MATMUL
_add = layer1.subgraph.add()
_add.name = "add"
_add.instance = dmir.ADD

_1 = layer1.dependency.add()
_1.operation = _matmul.name
_1.argument.append(_inp.name)
_1.argument.append(_wt.name)
_1.result.append(_accum.name)

_2 = layer1.dependency.add()
_2.operation = _add.name
_2.argument.append(_accum.name)
_2.argument.append(_bs.name)
_2.result.append(_out.name)

# layer2
_inp = layer2.input.add()
_inp.name = "input"
_inp.shape.append(1)
_inp.shape.append(512)
_inp.format = dmir.FLOAT

_wt = layer2.input.add()
_wt.name = "weight"
_wt.shape.append(512)
_wt.shape.append(512)
_wt.format = dmir.FLOAT

_bs = layer2.input.add()
_bs.name = "bias"
_bs.shape.append(512)
_bs.format = dmir.FLOAT

_out = layer2.output.add()
_out.name = "output"
_out.shape.append(1)
_out.shape.append(512)
_out.format = dmir.FLOAT

_accum = layer2.intermediate.add()
_accum.name = "product"
_accum.shape.append(1)
_accum.shape.append(512)
_accum.format = dmir.FLOAT

_matmul = layer2.subgraph.add()
_matmul.name = "matmul"
_matmul.instance = dmir.MATMUL
_add = layer2.subgraph.add()
_add.name = "add"
_add.instance = dmir.ADD

_1 = layer2.dependency.add()
_1.operation = _matmul.name
_1.argument.append(_inp.name)
_1.argument.append(_wt.name)
_1.result.append(_accum.name)

_2 = layer2.dependency.add()
_2.operation = _add.name
_2.argument.append(_accum.name)
_2.argument.append(_bs.name)
_2.result.append(_out.name)

# layer3
_inp = layer3.input.add()
_inp.name = "input"
_inp.shape.append(1)
_inp.shape.append(512)
_inp.format = dmir.FLOAT

_wt = layer3.input.add()
_wt.name = "weight"
_wt.shape.append(10)
_wt.shape.append(512)
_wt.format = dmir.FLOAT

_bs = layer3.input.add()
_bs.name = "bias"
_bs.shape.append(10)
_bs.format = dmir.FLOAT

_out = layer3.output.add()
_out.name = "output"
_out.shape.append(1)
_out.shape.append(10)
_out.format = dmir.FLOAT

_accum = layer3.intermediate.add()
_accum.name = "product"
_accum.shape.append(1)
_accum.shape.append(10)
_accum.format = dmir.FLOAT

_matmul = layer3.subgraph.add()
_matmul.name = "matmul"
_matmul.instance = dmir.MATMUL
_add = layer3.subgraph.add()
_add.name = "add"
_add.instance = dmir.ADD

_1 = layer3.dependency.add()
_1.operation = _matmul.name
_1.argument.append(_inp.name)
_1.argument.append(_wt.name)
_1.result.append(_accum.name)

_2 = layer3.dependency.add()
_2.operation = _add.name
_2.argument.append(_accum.name)
_2.argument.append(_bs.name)
_2.result.append(_out.name)

from google.protobuf.json_format import MessageToJson
f = open("lenet-512-512.json", "w")
f.write(MessageToJson(model))
f.close()