# Skipping tests as corsair aware cannot be removed at the moment
# # Contents of corsair aware must be commented out
# from platform import node
# import torch
# from mltools.fx.tracer import QuantTracer
# from mltools.fx.transform import cast_input_output_transform
# from mltools.fx.transformer import NodeDictTransformer
# from mltools.utils.fx.visualize_graph import GraphvizInterpreter
# from mltools.models import LeNet
# from mltools import corsair

# RANDOM_SEED = 0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(RANDOM_SEED)
# directory='./doctest-output/'

# def test_linear_without_transform(request):
#     test_name = request.node.name
#     net = torch.nn.Linear(64,64)
#     gm = torch.fx.symbolic_trace(net)
#     nodeDict = NodeDictTransformer(gm).transform()
#     gi = GraphvizInterpreter(gm,nodeDict)
#     inp = torch.rand(1,64)
#     gi.run(inp)
#     gi.pygraph.render(filename = directory+test_name)

# def test_linear(request):
#     test_name = request.node.name
#     net = torch.nn.Linear(64,64)
#     gm = cast_input_output_transform(net)
#     nodeDict = NodeDictTransformer(gm).transform()
#     gi = GraphvizInterpreter(gm,nodeDict)
#     inp = torch.rand(1,64)
#     gi.run(inp)
#     gi.pygraph.render(filename = directory+test_name)

# def test_linear_corsair(request):
#     test_name = request.node.name
#     net = corsair.nn.Linear(64,64)
#     tracer = QuantTracer()
#     graph = tracer.trace(net)
#     gm = torch.fx.GraphModule(tracer.root, graph)
#     nodeDict = NodeDictTransformer(gm).transform()
#     gi = GraphvizInterpreter(gm,nodeDict)
#     inp = torch.rand(1,64)
#     gi.run(inp)
#     gi.pygraph.render(filename = directory+test_name)

# def test_lenet_without_transform(request):
#     test_name = request.node.name
#     net = LeNet([10,10])
#     tracer = QuantTracer()
#     graph = tracer.trace(net)
#     gm = torch.fx.GraphModule(tracer.root, graph)
#     nodeDict = NodeDictTransformer(gm).transform()
#     gi = GraphvizInterpreter(gm,nodeDict)
#     inp = torch.rand(1,784)
#     gi.run(inp)
#     gi.pygraph.render(filename = directory+test_name)

# def test_lenet_without_cfg(request):
#     test_name = request.node.name
#     net = LeNet([10,10])
#     gm = cast_input_output_transform(net)
#     nodeDict = NodeDictTransformer(gm).transform()
#     gi = GraphvizInterpreter(gm,nodeDict)
#     inp = torch.rand(1,784)
#     gi.run(inp)
#     gi.pygraph.render(filename = directory+test_name)

# def test_lenet_with_cfg(request):
#     test_name = request.node.name
#     net = LeNet([10,10])
#     gm = cast_input_output_transform(net,cfg="../configs/lenet_test.yaml")
#     nodeDict = NodeDictTransformer(gm).transform()
#     gi = GraphvizInterpreter(gm,nodeDict)
#     inp = torch.rand(1,784)
#     gi.run(inp)
#     gi.pygraph.render(filename = directory+test_name)


# def test_dropout(request):
#     test_name = request.node.name
#     net = torch.nn.Dropout()
#     gm = cast_input_output_transform(net)
#     nodeDict = NodeDictTransformer(gm).transform()
#     gi = GraphvizInterpreter(gm,nodeDict)
#     inp = torch.rand(1,64)
#     gi.run(inp)
#     gi.pygraph.render(filename = directory+test_name)

# def test_AdaptiveAvgPool2d(request):
#     test_name = request.node.name
#     net = torch.nn.AdaptiveAvgPool2d(16)
#     gm = cast_input_output_transform(net)
#     nodeDict = NodeDictTransformer(gm).transform()
#     gi = GraphvizInterpreter(gm,nodeDict)
#     inp = torch.rand(1,64,64)
#     gi.run(inp)
#     gi.pygraph.render(filename = directory+test_name)


