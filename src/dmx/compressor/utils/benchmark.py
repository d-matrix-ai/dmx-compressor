import sys
import time
import torch
from collections import defaultdict
from dmx.compressor.modeling import DmxModel
import time
from dmx.compressor.functional import NoApproximation
from tabulate import tabulate
import enum
import gc

from typing import (
    Any,
    Dict,
    Tuple,
    Callable,
    List,
    Union,
)


class EVALUATION_MODE(enum.Enum):
    VANILLA = "Vanilla"
    BASELINE = "Baseline"
    FP8 = "fp8"
    BASIC = "Basic"
    BASIC_NOVSIMD = "Basic_NoVSIMD"


def prepare_model(
    model: torch.nn.Module,
    evaluation_mode: EVALUATION_MODE,
    model_runner: Callable[[Union[torch.nn.Module, DmxModel]], None],
):
    """Prepares a DMXModel if needed in Baseline, Basic, or Basic_NoVSIMD modes

    Parameters
    ----------
    model : torch.nn.Module
        torch model to prepare
    evaluation_mode : EVALUATION_MODE
        The mode for the DMXModel to create out of the torch model
    model_runner : Callable[[Union[torch.nn.Module, DmxModel]], None]
        A function for running a sample input through the model


    """
    vsimd_modules_by_type = defaultdict(list)

    if evaluation_mode == EVALUATION_MODE.BASELINE:
        model = DmxModel.from_torch(model)
        model.to_baseline_mode()
        model_runner(model)
    elif evaluation_mode == EVALUATION_MODE.FP8:
        model = DmxModel.from_torch(model)
        model.to_fp8_mode()
        model_runner(model)
    elif evaluation_mode in [EVALUATION_MODE.BASIC, EVALUATION_MODE.BASIC_NOVSIMD]:
        model = DmxModel.from_torch(model)
        model.to_basic_mode()
        model_runner(model)
        vsimd_modules = [
            (x[0], x[1])
            for x in model.named_dmx_modules()
            if not isinstance(x[1].approximator.function, NoApproximation)
        ]
        for name, m in vsimd_modules:
            if evaluation_mode == EVALUATION_MODE.BASIC_NOVSIMD:
                m.approximator.function = NoApproximation()
            vsimd_modules_by_type[type(m)].append(".".join(name.split(".")[1:]))

    return model, vsimd_modules_by_type


def measure_mode_perf(
    model: torch.nn.Module,
    model_runner: Callable[[Union[torch.nn.Module, DmxModel]], None],
    device: torch.device,
    evaluation_mode: EVALUATION_MODE,
    n_warmup_runs: int = 1,
    n_measure_runs: int = 5,
):
    """Measure model's runtime statistics for a given mode

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model to measure
    model_runner : Callable[[Union[torch.nn.Module, DmxModel]], None]
        Callable to run a sample input through the model
    device : torch.device
        device on which to run the model
    evaluation_mode : EVALUATION_MODE
        Model mode (Vanilla torch, Baseline, Basic without VSIMD ops,
        Basic)
    n_warmup_runs : int
        Number of warmup runs before gathering statistics
    n_measure_runs : int
        Number of runs across which to average the gathered statistics


    """
    assert n_warmup_runs >= 1 and n_measure_runs > 1
    model, vsimd_modules_by_type = prepare_model(model, evaluation_mode, model_runner)
    for _ in range(n_warmup_runs):
        model_runner(model)

    if evaluation_mode == EVALUATION_MODE.VANILLA:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(n_measure_runs):
            model_runner(model)
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        return {"total_time": (t2 - t1) / n_measure_runs, "per_layer_times": {}}
    else:

        mod_names = [
            ".".join(x[0].split(".")[1:]) for x in list(model.named_dmx_modules())
        ]
        all_runtimes = []
        for _ in range(n_measure_runs):
            with model.measure_runtimes(device, mod_names):
                model_runner(model)
            all_runtimes.append(model.get_runtime_records())

        average_perlayer_runtimes = {
            k: sum([sum(single_run_runtime[k]) for single_run_runtime in all_runtimes])
            / len(all_runtimes)
            for k in mod_names
        }

        total_time = sum(average_perlayer_runtimes.values())
        return {
            "total_time": total_time,
            "per_layer_times": average_perlayer_runtimes,
            "vsimd_modules_by_type": vsimd_modules_by_type,
        }


def evaluate_vsimd_ops_deltas(results: Dict[str, Any]):
    """Generate summary table with the runtime impact of each type of vsimd op

    Parameters
    ----------
    results : Dict[str, Any]
        The runtime measuerement results

    """
    b_val = EVALUATION_MODE.BASIC.value
    b_novsimd_val = EVALUATION_MODE.BASIC_NOVSIMD.value
    if b_val in results and b_novsimd_val in results:
        vsimd_modules_by_type = results[b_novsimd_val]["vsimd_modules_by_type"]
        vsimd_time_deltas = {}
        for type_name, layer_names in vsimd_modules_by_type.items():
            time_basic = sum(
                results[b_val]["per_layer_times"][lay_name] for lay_name in layer_names
            )
            time_basic_novsimd = sum(
                results[b_novsimd_val]["per_layer_times"][lay_name]
                for lay_name in layer_names
            )
            vsimd_time_deltas[type_name] = time_basic - time_basic_novsimd

        return vsimd_time_deltas

    print(
        "Specify EVALUATION_MODE.BASIC and EVALUATION_MODE.BASIC_NOVSIMD  \
    to obtain vsimd op deltas",
        list(results.keys()),
    )
    return None


def measure_model_runtime(
    model_maker: Callable[[], Tuple[torch.nn.Module, Callable, Callable, torch.device]],
    modes: List[EVALUATION_MODE],
):
    """Entry function for measuring various runtime statistics

    Parameters
    ----------
    model_maker : Callable[[], Tuple[torch.nn.Module, Callable, Callable, torch.device]]
        A callable that returns the model to be measured, together
        with some callables to run a sample input through the model or
        to evaluate the model's accuracy
    modes : List[EVALUATION_MODE]
        List of modes on which to evaluate the model's runtime
    """
    results = {}
    layer_names = []
    for mode in modes:
        print(f"Starting runtime measurements for mode {mode.value}")
        model, model_runner, _, device = model_maker()
        torch.cuda.reset_peak_memory_stats(device)
        results[mode.value] = measure_mode_perf(model, model_runner, device, mode)
        results[mode.value]["max_memory"] = torch.cuda.max_memory_allocated()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if len(results[mode.value]["per_layer_times"]) > len(layer_names):
            layer_names = list(results[mode.value]["per_layer_times"].keys())

    main_table_str = tabulate(
        {
            " ": ["max memory (GB)", "total time (s)", *layer_names],
            **{
                k: [
                    results[k]["max_memory"] / 2**30,
                    results[k]["total_time"],
                    *results[k]["per_layer_times"].values(),
                ]
                for k in results
            },
        },
        headers="keys",
        tablefmt="github",
    )
    print(main_table_str)

    vsimd_time_deltas = evaluate_vsimd_ops_deltas(results)
    if vsimd_time_deltas is not None:
        print("*********************VSIMD operations **********\n\n\n")
        lay_types = list(vsimd_time_deltas.keys())
        lay_deltas = list(vsimd_time_deltas.values())
        cumulative_runtime = (
            (
                torch.cumsum(torch.Tensor([0] + lay_deltas), dim=0)
                + results[EVALUATION_MODE.BASIC_NOVSIMD.value]["total_time"]
            )
            .numpy()
            .tolist()
        )

        vsimd_deltas_table_str = tabulate(
            {
                "Layer type": ["(BASIC mode time without VSIMD ops)"] + lay_types,
                "Time deltas": [" "] + lay_deltas,
                "Total run time": cumulative_runtime,
            },
            headers="keys",
            tablefmt="github",
        )

        print(vsimd_deltas_table_str)


def measure_model_accuracy(
    model_maker: Callable[[], Tuple[torch.nn.Module, Callable, Callable, torch.device]],
    modes: List[EVALUATION_MODE],
):
    """Entry function for measuring a model's accuracy across various modes

    Parameters
    ----------
    model_maker : Callable[[], Tuple[torch.nn.Module, Callable, Callable, torch.device]]
        A callable that returns the model to be measured, together
        with some callables to run a sample input through the model or
        to evaluate the model's accuracy
    modes : List[EVALUATION_MODE]
        List of modes on which to measure the model's accuracy

    """
    results = {}
    for mode in modes:
        print(f"Starting evaluation for mode {mode.value}")
        model, model_runner, evaluation_fn, device = model_maker()
        model, _ = prepare_model(model, mode, model_runner)

        results[mode.value] = evaluation_fn(model, device, mode.value)

    metric_names = list(list(results.values())[0].keys())
    main_table_str = tabulate(
        {
            " ": metric_names,
            **{k: [results[k][metric] for metric in metric_names] for k in results},
        },
        headers="keys",
        tablefmt="github",
    )
    print(main_table_str)


def collect_layer_activations(
    model_maker: Callable[[], Tuple[torch.nn.Module, Callable, Callable, torch.device]],
    mode: EVALUATION_MODE,
):
    """Collect the output activations for each named DMX module in the model for the given mode

    Parameters
    ----------
    model_maker : Callable[[], Tuple[torch.nn.Module, Callable, Callable, torch.device]]
        A callable that returns the model to be measured, together
        with some callables to run a sample input through the model or
        to evaluate the model's accuracy
    mode : EVALUATION_MODE
        The mode for which we want to collect the layer activations


    """
    model, model_runner, evaluation_fn, device = model_maker()
    model, _ = prepare_model(model, mode, model_runner)

    if mode == EVALUATION_MODE.VANILLA:
        with torch.no_grad():
            final_output = model_runner(model)
        return {}, {}, final_output
    else:
        mods = [".".join(x[0].split(".")[1:]) for x in model.named_dmx_modules()]
        mods_dict = {
            ".".join(x[0].split(".")[1:]): x[1] for x in model.named_dmx_modules()
        }
        with torch.no_grad():
            with model.monitoring(mods):
                final_output = model_runner(model)

        return mods_dict, model.get_monitoring_records(), final_output


def gather_tensors(
    tensor_collection: Union[torch.Tensor, List[Any], Tuple[Any], Dict[str, Any]],
) -> List[torch.Tensor]:
    """Gathers all torch tensors from arbitrary nested structures of Lists and Dicts

    Parameters
    ----------
    tensor_collection : Union[torch.Tensor, List[Any], Tuple[Any], Dict[str, Any]]
        A Torch tensor or an arbitrary collection of tensors such as
        what you would typically get as an output from a HuggingFace
        model

    Returns
    -------
    List[torch.Tensor]



    """
    if isinstance(tensor_collection, torch.Tensor):
        return [tensor_collection]
    elif isinstance(tensor_collection, (tuple, list)):
        return [tens for x in tensor_collection for tens in gather_tensors(x)]
    elif isinstance(tensor_collection, dict):
        return [tens for v in tensor_collection.values() for tens in gather_tensors(v)]
    else:
        return []


def compute_mse_error(
    t_list1: List[torch.Tensor], t_list2: List[torch.Tensor]
) -> float:
    """Compute sum of MSE errors between corresponding pairs of tensors in the given tensor lists

    Parameters
    ----------
    t_list1 : List[torch.Tensor]
    t_list2 : List[torch.Tensor]

    Returns
    -------
    float

    """

    return sum(
        torch.nn.functional.mse_loss(x.float(), y.float()).item()
        for x, y in zip(t_list1, t_list2)
    )


def compute_maxdelta_error(
    t_list1: List[torch.Tensor], t_list2: List[torch.Tensor]
) -> float:
    """Compute the maximum delta observed between two tensor elements across all pairs of tensors in the given lists

    Parameters
    ----------
    t_list1 : List[torch.Tensor]
    t_list2 : List[torch.Tensor]

    Returns
    -------
    float


    """
    return max(
        [(x - y).float().abs().max().item() for x, y in zip(t_list1, t_list2)] + [0]
    )


def compute_error(
    out1: Union[torch.Tensor, List[Any], Tuple[Any], Dict[str, Any]],
    out2: Union[torch.Tensor, List[Any], Tuple[Any], Dict[str, Any]],
):
    """Computes the MSE error and the maximum delta between the tensors in the given pair of tensor collections

    Parameters
    ----------
    out1 : Union[torch.Tensor, List[Any], Tuple[Any], Dict[str, Any]]
    out2 : Union[torch.Tensor, List[Any], Tuple[Any], Dict[str, Any]]


    """
    t_list1 = gather_tensors(out1)
    t_list2 = gather_tensors(out2)
    return {
        "mse": compute_mse_error(t_list1, t_list2),
        "maxdelta": compute_maxdelta_error(t_list1, t_list2),
    }


def measure_model_error(
    model_maker: Callable[[], Tuple[torch.nn.Module, Callable, Callable, torch.device]],
    modes: List[EVALUATION_MODE],
    reference_mode: EVALUATION_MODE = EVALUATION_MODE.BASELINE,
):
    """Entry function for measuring the error at the output of each layer of the model for each mode relative that layer's output in the reference mode

    Parameters
    ----------
    model_maker : Callable[[], Tuple[torch.nn.Module, Callable, Callable, torch.device]]
        A callable that returns the model to be measured, togetherwith
        some callables to run a sample input through the model or to
        evaluate the model's accuracy
    modes : List[EVALUATION_MODE]
        The modes for which we want to evaluate the errors in the
        layer's activations
    reference_mode : EVALUATION_MODE
        The reference mode whose layer activations serve as the ground
        truth used to evaluate the per-layer errors for the other
        modes


    """

    print(f"collecting activations for reference {reference_mode.value}")
    _, reference_activations, reference_final_output = collect_layer_activations(
        model_maker, reference_mode
    )
    results = defaultdict(dict)
    table_lay_names = []
    for mode in modes:
        if mode == reference_mode:
            continue
        print(f"collecting activations for {mode.value}")
        mods_dict, activations, final_output = collect_layer_activations(
            model_maker, mode
        )
        if activations and reference_activations:
            lay_names = list(mods_dict.keys())
            cum_error_dict = defaultdict(lambda: defaultdict(float))
            per_layer_error_dict = defaultdict(lambda: defaultdict(float))
            input_error_dict = defaultdict(lambda: defaultdict(float))
            for l_name in lay_names:
                assert len(activations[l_name]) == len(reference_activations[l_name])
                for datum1, datum2 in zip(
                    reference_activations[l_name], activations[l_name]
                ):
                    for metric, val in compute_error(
                        datum1["input"], datum2["input"]
                    ).items():
                        input_error_dict[l_name][metric] += val

                    out1 = datum1["output"]
                    out2 = datum2["output"]
                    for metric, val in compute_error(out1, out2).items():
                        cum_error_dict[l_name][metric] += val

                    input_args = datum1["input"][0]
                    input_kwargs = datum1["input"][1]
                    model2_clean_input_output = mods_dict[l_name](
                        *input_args, **input_kwargs
                    )

                    for metric, val in compute_error(
                        out1, model2_clean_input_output
                    ).items():
                        per_layer_error_dict[l_name][metric] += val

            results[mode.value] = {
                "cumulative": cum_error_dict,
                "per_layer": per_layer_error_dict,
                "input": input_error_dict,
            }
            if len(lay_names) > len(table_lay_names):
                table_lay_names = lay_names

        results[mode.value]["final_output_error"] = compute_error(
            final_output, reference_final_output
        )

        del activations, final_output
        gc.collect()
        torch.cuda.empty_cache()

    def make_error_str(error_dict):
        return f'{error_dict["mse"]:.2g}({error_dict["maxdelta"]:.2g})'

    table_dict = {}
    error_types = ["per_layer", "cumulative", "input"]
    error_format = "mse(max delta)"
    for k in results:
        for e_type in error_types:
            if e_type in results[k]:
                table_dict[f"{k}({e_type})"] = [
                    error_format,
                    make_error_str(results[k]["final_output_error"]),
                ] + [
                    make_error_str(results[k][e_type][l_name])
                    for l_name in table_lay_names
                ]
        if not any(e_type in results[k] for e_type in error_types):
            table_dict[k] = [
                error_format,
                make_error_str(results[k]["final_output_error"]),
            ]
    table_str = tabulate(
        {
            f"error relative to {reference_mode.value}": [
                "error_format",
                "final_output_error",
            ]
            + table_lay_names,
            **table_dict,
        },
        headers="keys",
        tablefmt="github",
    )

    print(table_str)
