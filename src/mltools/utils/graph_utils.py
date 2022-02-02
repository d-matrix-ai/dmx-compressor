from mltools import utils


def graph_dict_to_csv(graph_dict):
    csv_str = "LyrName,Type,Stride,OutT(W),OutT(H),OutT(D),OutT(Size),InT(W),InT(H),InT(D),InT(Size),WgtT(W),WgtT(H),WgtT(D),WgtT(Num),WgtT(Size),Total MAC ops,Exp MAC Util,InType,WgtType,OutType"

    for layer_name, layer_attrs in graph_dict.items():
        if layer_attrs["op"] == "conv":
            # conv1_t,Convolution,2,112,112,64,802816,224,224,3,150528,7,7,3,64,9408,118013952,0.3,BFP16,BFP16,BFP16
            line = f"\n{layer_name},"
            line += f"Convolution,"
            line += f"{layer_attrs['stride']},"
            line += f"{layer_attrs['output_shape'][2]},{layer_attrs['output_shape'][3]},{layer_attrs['output_shape'][1]},,"
            line += f"{layer_attrs['input_shape'][2]},{layer_attrs['input_shape'][3]},{layer_attrs['input_shape'][1]},,"
            line += f"{layer_attrs['param_shape'][2]},{layer_attrs['param_shape'][3]},{layer_attrs['param_shape'][1]},{layer_attrs['param_shape'][0]},,,,"
            line += f"{layer_attrs['input_format']},{layer_attrs['param_format']},{layer_attrs['output_format']}"
            csv_str += line
        elif layer_attrs["op"] == "linear" or layer_attrs["op"] == "matmul":
            # matmul64x64,Convolution,1,1,1,64,64,1,1,64,64,1,1,64,64,4096,,,BFP16,BFP16,BFP16
            line = f"\n{layer_name},"
            line += f"Convolution,"
            line += f"1,1,1,{layer_attrs['output_shape'][1]},,"
            line += f"1,1,{layer_attrs['input_shape'][1]},,"
            line += f"1,1,{layer_attrs['param_shape'][1]},{layer_attrs['param_shape'][0]},,,,"
            line += f"{layer_attrs['input_format']},{layer_attrs['param_format']},{layer_attrs['output_format']}"
            csv_str += line
        elif layer_attrs["op"] == "max_pool":
            # pool1_t,MaxPooling,2,56,56,64,200704,112,112,64,802816,,,,,0,0,0.3,BFP16,,BFP16
            line = f"\n{layer_name},"
            line += f"MaxPooling,"
            line += f"{layer_attrs['stride']},"
            line += f"{layer_attrs['output_shape'][2]},{layer_attrs['output_shape'][3]},{layer_attrs['output_shape'][1]},,"
            line += f"{layer_attrs['input_shape'][2]},{layer_attrs['input_shape'][3]},{layer_attrs['input_shape'][1]},,,,,,0,0,,"
            line += f"{layer_attrs['input_format']},,{layer_attrs['output_format']}"
            csv_str += line
        elif layer_attrs["op"] == "adaptive_average_pool":
            # pool5_t,AvgPooling,1,1,1,2048,2048,7,7,2048,100352,,,,,0,0,0.3,BFP16,,BFP16
            line = f"\n{layer_name},"
            line += f"AvgPooling,"
            line += f"1,1,1,{layer_attrs['output_shape'][1]},,"
            line += f"{layer_attrs['input_shape'][2]},{layer_attrs['input_shape'][3]},{layer_attrs['input_shape'][1]},,,,,,0,0,,"
            line += f"{layer_attrs['input_format']},,{layer_attrs['output_format']}"
            csv_str += line
        elif layer_attrs["op"] == "relu":
            # conv1_relu_t,ReLU,1,112,112,64,802816,112,112,64,802816,,,,,0,0,0.3,BFP16,,BFP16
            line = f"\n{layer_name},"
            line += f"ReLU,"
            if len(layer_attrs["output_shape"]) > 2:
                line += f"1,{layer_attrs['output_shape'][2]},{layer_attrs['output_shape'][3]},{layer_attrs['output_shape'][1]},,"
                line += f"{layer_attrs['output_shape'][2]},{layer_attrs['output_shape'][3]},{layer_attrs['output_shape'][1]},,,,,,0,0,,"
            else:
                line += f"1,1,1,{layer_attrs['output_shape'][1]},,"
                line += f"1,1,{layer_attrs['output_shape'][1]},,,,,,0,0,,"
            line += f"{layer_attrs['output_format']},,{layer_attrs['output_format']}"
            csv_str += line
        elif layer_attrs["op"] == "add":
            # res2a,Eltwise Add,1,56,56,256,802816,56,56,256,802816,56,56,256,,0,0,0.3,BFP16,,BFP16
            line = f"\n{layer_name},"
            line += f"Eltwise Add,"
            line += f"1,{layer_attrs['output_shape'][2]},{layer_attrs['output_shape'][3]},{layer_attrs['output_shape'][1]},,"
            line += f"{layer_attrs['output_shape'][2]},{layer_attrs['output_shape'][3]},{layer_attrs['output_shape'][1]},,"
            line += f"{layer_attrs['output_shape'][2]},{layer_attrs['output_shape'][3]},{layer_attrs['output_shape'][1]},,0,0,,"
            line += f"{layer_attrs['output_format']},,{layer_attrs['output_format']}"
            csv_str += line

    return csv_str


def save_to_file(graph_dict: dict, filename: str, format="csv") -> None:
    if format == "csv":
        csv_str = utils.graph_dict_to_csv(graph_dict)
        with open(filename, "w") as f:
            f.write(csv_str)
    elif format == "json":
        import json

        json_dict = json.dumps(graph_dict, indent=4)
        with open(filename, "w") as f:
            f.write(json_dict)
    else:
        raise ValueError(f"unsupported Graph file format: {format}")


def prune_graph(graph_dict, ops_to_prune=("flatten", "batchnorm", "reshape")):
    # remove specified ops from graph, overwrite inputs/outputs for affected nodes
    found = False
    # find the first node to be eliminated:
    for node in graph_dict:
        if graph_dict[node]["op"] in ops_to_prune:
            # assume this node has only one input
            input_node = graph_dict[node]["input_name"][0]
            # find all nodes that use this node as input
            for n in graph_dict:
                # node can have multiple inputs, check all of them
                for i, name in enumerate(graph_dict[n]["input_name"]):
                    if name == node:
                        graph_dict[n]["input_name"][i] = input_node
                        found = True

            # do only one removal per graph traversal
            if found:
                del graph_dict[node]
                break
    if not found:
        # End of pruning, now remove input and output nodes
        node_names = []
        for node in graph_dict:
            if graph_dict[node]["op"] in ["input", "output"]:
                node_names.append(node)
        for node in node_names:
            del graph_dict[node]
        return graph_dict
    else:
        return prune_graph(graph_dict)
