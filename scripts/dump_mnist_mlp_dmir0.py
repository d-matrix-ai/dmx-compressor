from os import name
import utils.dmir_pb2 as dmir

layer1 = dmir.Graph(
    name="input_layer",
    input=(
        dmir.Tensor(name="input", shape=(1, 784), format=dmir.FLOAT),
        dmir.Tensor(name="weight", shape=(512, 784), format=dmir.FLOAT),
        dmir.Tensor(name="bias", shape=(512,), format=dmir.FLOAT),
    ),
    output=(dmir.Tensor(name="output", shape=(1, 512), format=dmir.FLOAT),),
    intermediate=(dmir.Tensor(name="product", shape=(1, 512), format=dmir.FLOAT),),
    subgraph=(
        dmir.Graph(
            name="matmul",
            instance=dmir.MATMUL,
        ),
        dmir.Graph(
            name="add",
            instance=dmir.ADD,
        ),
    ),
    dependency=(
        dmir.Dependency(
            operation="matmul",
            argument=("input", "weight"),
            result=("product",),
        ),
        dmir.Dependency(
            operation="add",
            argument=("product", "bias"),
            result=("output",),
        ),
    ),
)

layer2 = dmir.Graph(
    name="intermediate_layer.0",
    input=(
        dmir.Tensor(name="input", shape=(1, 512), format=dmir.FLOAT),
        dmir.Tensor(name="weight", shape=(512, 512), format=dmir.FLOAT),
        dmir.Tensor(name="bias", shape=(512,), format=dmir.FLOAT),
    ),
    output=(dmir.Tensor(name="output", shape=(1, 512), format=dmir.FLOAT),),
    intermediate=(dmir.Tensor(name="product", shape=(1, 512), format=dmir.FLOAT),),
    subgraph=(
        dmir.Graph(
            name="matmul",
            instance=dmir.MATMUL,
        ),
        dmir.Graph(
            name="add",
            instance=dmir.ADD,
        ),
    ),
    dependency=(
        dmir.Dependency(
            operation="matmul",
            argument=("input", "weight"),
            result=("product",),
        ),
        dmir.Dependency(
            operation="add",
            argument=("product", "bias"),
            result=("output",),
        ),
    ),
)

layer3 = dmir.Graph(
    name="output_layer",
    input=(
        dmir.Tensor(name="input", shape=(1, 512), format=dmir.FLOAT),
        dmir.Tensor(name="weight", shape=(10, 512), format=dmir.FLOAT),
        dmir.Tensor(name="bias", shape=(10,), format=dmir.FLOAT),
    ),
    output=(dmir.Tensor(name="output", shape=(1, 512), format=dmir.FLOAT),),
    intermediate=(dmir.Tensor(name="product", shape=(1, 10), format=dmir.FLOAT),),
    subgraph=(
        dmir.Graph(
            name="matmul",
            instance=dmir.MATMUL,
        ),
        dmir.Graph(
            name="add",
            instance=dmir.ADD,
        ),
    ),
    dependency=(
        dmir.Dependency(
            operation="matmul",
            argument=("input", "weight"),
            result=("product",),
        ),
        dmir.Dependency(
            operation="add",
            argument=("product", "bias"),
            result=("output",),
        ),
    ),
)

act_fn1 = dmir.Graph(
    name="act_fn1",
    instance=dmir.RELU,
)

act_fn2 = dmir.Graph(
    name="act_fn2",
    instance=dmir.RELU,
)

model = dmir.Graph(
    name="lenet-512-512",
    input=(dmir.Tensor(name="flat_images", shape=(1, 784), format=dmir.FLOAT),),
    output=(dmir.Tensor(name="logits", shape=(1, 10), format=dmir.FLOAT),),
    intermediate=(
        dmir.Tensor(name="v0", shape=(1, 512), format=dmir.FLOAT),
        dmir.Tensor(name="v1", shape=(1, 512), format=dmir.FLOAT),
        dmir.Tensor(name="v2", shape=(1, 512), format=dmir.FLOAT),
        dmir.Tensor(name="v3", shape=(1, 512), format=dmir.FLOAT),
    ),
    subgraph=(
        layer1,
        layer2,
        layer3,
        act_fn1,
        act_fn2,
    ),
    dependency=(
        dmir.Dependency(
            operation=layer1.name,
            argument=("flat_images",),
            result=("v0",),
        ),
        dmir.Dependency(
            operation=act_fn1.name,
            argument=("v0",),
            result=("v1",),
        ),
        dmir.Dependency(
            operation=layer2.name,
            argument=("v1",),
            result=("v2",),
        ),
        dmir.Dependency(
            operation=act_fn2.name,
            argument=("v2",),
            result=("v3",),
        ),
        dmir.Dependency(
            operation=layer2.name,
            argument=("v3",),
            result=("logits",),
        ),
    ),
)


from google.protobuf.json_format import MessageToJson

with open("lenet-512-512.dmir", "w") as f:
    f.write(MessageToJson(model))
