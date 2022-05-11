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
    
def prune_graph(graph_dict, childless_nodes=None, debug=False, ops_to_prune=("flatten", 
                                                                            "batchnorm",
                                                                            "layer_norm",
                                                                            "dropout",
                                                                            "truediv",
                                                                            "reshape", 
                                                                            "size", 
                                                                            "view", 
                                                                            "split", 
                                                                            "squeeze",
                                                                            "expand",
                                                                            "getitem",
                                                                            "permute",
                                                                            "clone",
                                                                            "identity",
                                                                            "contiguous",
                                                                            "transpose",
                                                                            "t",
                                                                            "ones",
                                                                            "to",
                                                                            "sub",
                                                                            "mul",
                                                                            "attention_block",
                                                                            )):

    # remove specified ops from graph, overwrite inputs/outputs for affected nodes
    if childless_nodes is None:  # only print the graph once (the first iteration)
        if debug:
            print('\n\n\n\n\nUnpruned graph\n\n\n')
        for node in graph_dict:
            if debug:
                print(node, graph_dict[node]["op"], [i for i in graph_dict[node]["input_name"]])
        
        # find detached nodes (nodes which are not inputs to any other nodes in the graph)
        if debug:
            print('\n\n\n\nLooking for childless nodes\n\n')
        childless_nodes = []
        for node in graph_dict:
            found_as_input = False
            for n in graph_dict:
                if node in graph_dict[n]["input_name"]:
                    found_as_input = True
            if not found_as_input:
                if debug:
                    print(f'\n\n{node}: {graph_dict[node]}')
                childless_nodes.append(node)
                
    found = False
    # find the first node to be eliminated:
    for node in graph_dict:
        if 'split' in graph_dict[node]["op"] and debug:
            if debug:
                print('\n\n\n\n', node, '\n\n\n', graph_dict[node], '\n\n\n')

        if ((graph_dict[node]["op"] in ops_to_prune 
            or any([name in graph_dict[node]["op"] for name in ["layer_norm", 
                                                                "embedding", 
                                                                "cast", 
                                                                "transpose_for_scores",
                                                                "view_for_context"
                                                                ]])
            or 'add' in node and graph_dict[node]["input_name"][1][0] =="(") # second input is a tuple
            and graph_dict[node]["input_name"] != []):
            # assume this node has only one input TODO what about multi input node deletion? It might create childless nodes
            # for example, "ones" op in bert_large takes ['getitem', 'add'] as inputs, and deleting it makes  "add" childless
            input_node = graph_dict[node]["input_name"][0]
            # find all nodes that use this node as input
            for n in graph_dict:
                # node can have multiple inputs, check all of them
                for i, name in enumerate(graph_dict[n]["input_name"]):
                    if name == node:
                        #if debug:
                            #print(f'\n\n{node} is input of {n}, deleting {node} and assigning its input {input_node} to be input of {n}\n\n')
                        graph_dict[n]["input_name"][i] = input_node
                        found = True

            # do only one removal per graph traversal
            if found:
                del graph_dict[node]
                break

    if not found:
        # End of pruning, now remove any leaf nodes
        if debug:
            print('\n\n\n\nPartially pruned graph\n\n\n')
        for node in graph_dict:
            if debug:
                print(node, graph_dict[node]["op"], [i for i in graph_dict[node]["input_name"]])

        if debug:
            print('\n\n\n\nLooking for childless nodes (nodes which are not inputs to any other node) \n\n')
        childless_nodes_after_pruning = []
        for node in graph_dict:
            found_as_input = False
            for n in graph_dict:
                if node in graph_dict[n]["input_name"]:
                    found_as_input = True
            if not found_as_input:
                if debug:
                    print(f'\n\n{node}: {graph_dict[node]}\n\n')
                if graph_dict[node]["op"] == "attention_block":
                    if debug:
                        print(f'*** ignoring {node} - not a real node ***')
                elif node == "add":
                    if debug:
                        print(f'*** ignoring {node} - extra input to "ones" ***')
                else:
                    childless_nodes_after_pruning.append(node)
                
        # check that no new childless nodes have been created during pruning:
        if debug:
            print(f'\n\nChecking that no childless nodes got created during pruning:\n\nBefore:\n'
                  f'{childless_nodes}\nAfter:\n{childless_nodes_after_pruning}')
        assert(sorted(childless_nodes) == sorted(childless_nodes_after_pruning))
        
        if debug:
            print('\n\n\n\nLooking for leaf nodes (nodes which have no inputs)\n\n')
        leaf_nodes = []
        for node in graph_dict:
            if graph_dict[node]["input_name"] == []:
                if debug:
                    print(f'\n\n{node}: {graph_dict[node]}')
                leaf_nodes.append(node)
                
        nodes_to_remove = set(childless_nodes + leaf_nodes)
        for node in nodes_to_remove:
            del graph_dict[node]
            
        if debug:
            print('\n\n\n\nFinal pruned graph:\n\n') 
        for node in graph_dict:
            if debug:
                print(node, graph_dict[node]["op"], [i for i in graph_dict[node]["input_name"]])
        if debug:
            print('\n\n\n\n\n')
        return graph_dict
    else:
        return prune_graph(graph_dict, childless_nodes=childless_nodes, debug=debug)
