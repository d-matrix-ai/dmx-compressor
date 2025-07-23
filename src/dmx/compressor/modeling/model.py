import torch
import re
import warnings
from collections import OrderedDict
from inspect import signature, _empty
from types import SimpleNamespace
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import (
    Any,
    Dict,
    Optional,
    Union,
    Sequence,
    get_args,
    get_origin,
    get_type_hints,
)
from functools import partial
from dmx.compressor.modeling.nn import *
from dmx.compressor.fx.transform import (
    substitute_transform,
    prepare_tracing_inputs,
    make_compiler_graph,
)
from dmx.compressor.fx.transformer import get_op_set_from
from dmx.compressor.utils.fx.visualize_graph import visualize_graph


class DmxModelMixin:
    transformed: bool
    additional_dmx_aware_mappings: Dict
    _gm: Optional[torch.fx.GraphModule]  # current gm
    _gms: Dict  # stores {sig: gm} pairs
    _dmx_configuration_queue: List  # stores (config, rules) to be applied
    _monitoring_records: Optional[Dict]  # stored monitored submodule inputs/outputs
    _runtime_records: Optional[Dict]  # stored monitored submodule run times

    def _apply_config(self, config: Optional[Union[dict, str]], *rules):
        if config is not None:
            if isinstance(config, str):
                config = DmxConfig.from_yaml(config)

            for n, m in self.named_dmx_modules():
                if n in config:
                    m.configure(config[n])

        for _r in rules:
            _r.apply_to(self)

    def configure(self, config: Optional[Union[dict, str]], *rules):
        r"""
        Configure Dmx-specific numerics/sparsity/logics

        NOTE: only staticly declared DmxModule(s) are to be transformed

        Args:
            config (Optional[Union[DmxConfig, str]]): DmxConfig to be used for transformation.
            *rules (List[DmxConfigRule]): variable length of list of configuration rules on top of config.

        Returns:
            Returns the transformed model

        """
        self._dmx_configuration_queue.append((config, rules))
        if self.transformed:
            self._apply_config(config, *rules)
        return self

    transform = configure  # NOTE: to be deprecated

    @property
    def op_set(self):
        r"Returns a set of unique ops present in the model"
        return set.union(*[get_op_set_from(_m) for _m in self._gms.values()])

    @property
    def dmx_config(self):
        r""" "Returns the DmxConfig object for the model"""
        return DmxConfig.from_model(self, freeze=True)

    @property
    def dmx_module_names(self):
        r""" "Returns a list of module names listed in a dmx_config"""
        return self.dmx_config.module_names

    def named_dmx_modules(self):
        r""" "Returns a list of named modules that are dmx configurable"""
        all_modules = []
        for gm in self._gms.values():
            new_modules = [(f'_gm.{n}',m) for n,m in gm.named_modules() if \
                           is_configurable(m) and \
                           n not in [x[0] for x in all_modules]]
            all_modules.extend(new_modules)
        return tuple(all_modules)


    def freeze(self, config_file="./config.yaml"):
        """
        A function that stores the state and ops format of the model to a config file

        Args:
            config_file (Optional[str]): Path to store the config file. Defaults to "./config.yaml".
        """
        DmxConfig.from_model(self, freeze=True).to_yaml(config_file)

    def thaw(self, config_file="./config.yaml"):
        """
        A function that transforms the model in place from a config file.

        Args:
            config_file (Optional[str]): Path of config file to transform the model from. Defaults to "./config.yaml".
        """
        self.configure(config_file)

    def print_model_tree(self, include_type=False):
        """
        A function that prints out the tree structure of a model

        Args:
            include_type (bool): include the type of modules in the print out if True
        """
        from dmx.compressor.utils.visualization import print_model_tree

        print_model_tree(self, include_type)

    def fold_weights_and_biases(self):
        """
        A function that applies the ops the weights and biases using the corresponding formats.
        """
        for _, m in self.named_dmx_modules():
            m.fold_weight_and_bias()

    def check_dim_consistency(self) -> bool:
        """
        A function that checks format dimension consistency and sparseness dimension consistency for all applicable dmx modules in the model

        Returns:
            True if all dimensions are consistent for all applicable modules.
        """
        return all(
            m.check_format_dim_consistency() and m.check_sparseness_dim_consistency()
            for _, m in self.named_dmx_modules()
        )

    def to_baseline_mode(self):
        from dmx.compressor import config_rules

        # we can clear the _dmx_configuration_queue as baseline config rule will overwrite everything
        self._dmx_configuration_queue = []
        self.configure(None, *config_rules.BASELINE)

    def to_basic_mode(self, sbfp_weight_storage=False):
        """
        Configures a transformed DmxModel to the BASIC mode on dmx hardware.

        Returns:
            The configured model.
        """
        from dmx.compressor import config_rules

        self.to_baseline_mode()
        self.configure(None, *config_rules.BASIC)
        if sbfp_weight_storage:
            self.configure(None, *config_rules.SBFP_WEIGHT_STORAGE)
            self.forward_weight_hypernets()

    @contextmanager
    def keep_dmx_config(self):
        _old_dmx_config_queue = deepcopy(self._dmx_configuration_queue)
        yield self
        for _config, _rules in _old_dmx_config_queue:
            self._apply_config(_config, *_rules)
        self._dmx_configuration_queue = _old_dmx_config_queue

    @contextmanager
    def counting_flops(self, zero: bool = True):
        with ExitStack() as stack:
            yield [
                stack.enter_context(m.counting_flops(zero))
                for _, m in self.named_dmx_modules()
            ]

    @staticmethod
    def _save_specific_layers_state_dict_and_register_urls(
        specific_layers: Sequence[DmxModule],
        save_checkpoint_to: Optional[str],
    ):
        if save_checkpoint_to is not None:
            for _m in specific_layers:
                _m.save_state_dict_and_register_url(parent_dir=save_checkpoint_to)

    @contextmanager
    def monitoring(
        self,
        submodules_to_monitor: List[str] = [],
        save_checkpoint_to: Optional[str] = None,
    ):
        self._monitoring_records = {_sm: [] for _sm in submodules_to_monitor}
        with ExitStack() as stack:
            yield [
                stack.enter_context(
                    self._gm.get_submodule(_sm).monitoring(
                        self._monitoring_records[_sm]
                    ),
                )
                for _sm in submodules_to_monitor
            ]

    def get_monitoring_records(self, submodules_to_monitor: List[str] = []):
        _rec = self._monitoring_records
        self._monitoring_records = None

        return _rec

    @contextmanager
    def measure_runtimes(self, device, submodules_to_measure: List[str] = []):
        self._runtime_records = {_sm: [] for _sm in submodules_to_measure}
        with ExitStack() as stack:
            yield [
                stack.enter_context(
                    self._gm.get_submodule(_sm).measuring_runtime(
                        self._runtime_records[_sm], device
                    ),
                )
                for _sm in submodules_to_measure
            ]

    def get_runtime_records(self):
        _rec = self._runtime_records
        self._runtime_records = None

        return _rec

    @staticmethod
    def _add_transformed_gm(_model, args, kwargs):
        if hasattr(_model, "old_forward"):
            _model.forward = _model.old_forward
        input_names, concrete_args, dummy_inputs = prepare_tracing_inputs(
            _model, args, kwargs
        )
        _model._gm = substitute_transform(
            _model,
            input_names=input_names,
            concrete_args=concrete_args,
            dummy_inputs=dummy_inputs,
            additional_mappings=_model.additional_dmx_aware_mappings,
        )
        DmxModel.post_process_gm(_model, args, kwargs)

    @staticmethod
    def _get_transformed_forward(_model):
        if hasattr(_model, "old_forward"):
            _model.forward = _model.old_forward
        _mod_signature = signature(_model.forward)
        _output_cls = _mod_signature.return_annotation
        if get_origin(_output_cls) is Union:  # NOTE: this is error-prone
            transformer_output_cls = None
            for output_type in get_args(_output_cls):
                # filter out typing classes, eg: typing.Tuple, as they will throw error with issubclass
                if not hasattr(output_type, "__origin__") and issubclass(
                    output_type, transformers.modeling_utils.ModelOutput
                ):
                    transformer_output_cls = output_type
                    break
            _output_cls = transformer_output_cls
        elif (
            _output_cls is _empty
            or hasattr(_output_cls, "__origin__")
            or not issubclass(_output_cls, transformers.modeling_utils.ModelOutput)
        ):
            _output_cls = None

        _model._output_cls = _output_cls
        _forward = (
            (
                lambda *_args, **_kwargs: DmxModel._wrap_output_cls(
                    _model._gm.forward(*_args, **_kwargs), _output_cls
                )
            )
            if _model._output_cls is not None
            else _model._gm.forward
        )
        return _forward

    @staticmethod
    def _wrap_output_cls(outputs, cls):
        attribute_types = get_type_hints(cls)
        for key, expected_type in attribute_types.items():
            if key in outputs:
                if (
                    expected_type
                    and isinstance(expected_type, type)
                    and issubclass(
                        expected_type, transformers.utils.generic.ModelOutput
                    )
                ):
                    outputs[key] = DmxModel._wrap_output_cls(
                        outputs[key], expected_type
                    )  # Recursively wrap sub-objects

        return cls(**outputs)

    @staticmethod
    def is_same_signature(_model, args, kwargs):

        tracing_args, tracing_kwargs = _model.tracing_kwargs
        # if kwargs has different keys, need to retrace
        if tracing_kwargs.keys() != kwargs.keys():
            return False
        # if args has different length, need to retrace
        if len(tracing_args) != len(args):
            return False
        # comparing args values between tracing args and new args
        for a, new_a in zip(tracing_args, args):
            # if bool argument has different values, need to retrace
            if isinstance(new_a, bool) and a != new_a:
                return False
            # if one in None and other is not none, need to retrace
            if not isinstance(new_a, bool) and (new_a is None) != (a is None):
                return False
        # comparing kwargs values between tracing kwarg and new kwargs
        for k in kwargs.keys():
            # if bool argument has different values, need to retrace
            if isinstance(kwargs[k], bool) and kwargs[k] != tracing_kwargs[k]:
                return False
            # if type of argument does not match, need to retrace
            if type(kwargs[k]) != type(tracing_kwargs[k]):
                return False
            # if kwargs is DynamicCache and shape is different (i.e: one is empty), need to retrace.
            # NOTE: to be deprecated
            if isinstance(kwargs[k], transformers.DynamicCache) and len(
                kwargs[k]
            ) != len(tracing_kwargs[k]):
                return False
        return True

    def forward_weight_hypernets(self):
        for _, _m in self.named_dmx_modules():
            if hasattr(_m, "weight"):
                _ = _m._weight

    @staticmethod
    def deepcopy_args(args):
        def deepcopy_element(element):
            if isinstance(element, torch.Tensor):
                return element.detach().clone()
            elif isinstance(element, tuple):
                return type(element)(deepcopy_element(e) for e in element)
            elif isinstance(element, list):
                return type(element)([deepcopy_element(e) for e in element])
            elif isinstance(element, dict):
                return type(element)(
                    {k: deepcopy_element(v) for k, v in element.items()}
                )
            else:
                return deepcopy(element)

        return deepcopy_element(args)

    @staticmethod
    def post_process_gm(_model, args, kwargs):
        # some inputs were removed from input names due to None or bool, we want to add it back to maintain original input signature
        placeholders_needed = list(
            signature(_model.forward).bind(*args, **kwargs).arguments.keys()
        )
        node_list = _model._gm.graph.nodes
        i = 0
        for node in node_list:
            if i >= len(placeholders_needed):
                break
            while (
                i < len(placeholders_needed) and node.target != placeholders_needed[i]
            ):
                with _model._gm.graph.inserting_before(node):
                    _model._gm.graph.placeholder(placeholders_needed[i])
                    i += 1
            i += 1
        _model._gm.recompile()

    @classmethod
    def create_submod_transform_forward(
        cls,
        model: torch.nn.Module,
        submod_name: str,
        additional_dmx_aware_mappings=None,
    ):
        submod = model.get_submodule(submod_name)
        if DmxModelMixin not in submod.__class__.__bases__:
            submod.__class__.__bases__ += (DmxModelMixin,)
        submod._gm = None
        submod.transformed = False
        submod.config = model.config
        submod.additional_dmx_aware_mappings = additional_dmx_aware_mappings

        def temp_forward(_m, *_args, **_kwargs):
            _is_training = _m.training
            if not _m.transformed or not DmxModel.is_same_signature(_m, _args, _kwargs):
                if not model.transformed:
                    raise Exception(
                        "model forward needs to be called before submodule forward."
                    )
                warnings.warn("Submodule transformation triggered")
                _m.tracing_kwargs = (
                    DmxModel.deepcopy_args(_args),
                    DmxModel.deepcopy_args(_kwargs),
                )
                # because some args and kwargs are changed in place in forward, we need to keep a copy fo the _args and _kwargs
                forward_args, forward_kwargs = DmxModel.deepcopy_args(
                    _args
                ), DmxModel.deepcopy_args(_kwargs)

                DmxModel._add_transformed_gm(_m, _args, _kwargs)
                _m._forward = DmxModel._get_transformed_forward(_m)
                _m.transformed = True

                for n, _ in _m.named_dmx_modules():
                    dmx_mod_name = n[4:]  # first 4 chars are _gm.
                    dmx_mod_name = "_gm." + submod_name + "." + dmx_mod_name

                    # handling nested attr for setattr
                    pre, _, post = n.rpartition(".")
                    if pre:
                        parent = submod.get_submodule(pre)
                    else:
                        parent = submod
                    setattr(parent, post, model.get_submodule(dmx_mod_name))

                _m.train(_is_training)
                _m.transformed_forward = partial(temp_forward, _m)
                return _m._forward(*forward_args, **forward_kwargs)

            return _m._forward(*_args, **_kwargs)

        submod.old_forward = submod.forward
        submod.transformed_forward = partial(temp_forward, submod)

    @staticmethod
    def to_signature_key(_m, _args, _kwargs):
        sig = signature(_m.old_forward)
        inputs = signature(_m.old_forward).bind(*_args, **_kwargs).arguments
        sig_key = tuple()
        for k, v in sig.parameters.items():
            if k not in inputs:
                value = v.default
            else:
                value = inputs[k]
            if isinstance(value, bool) or value is None:
                sig_key += (value,)
            elif isinstance(value, transformers.DynamicCache) and len(value) == 0:
                sig_key += (None,)
            else:
                sig_key += (repr(type(value)),)
        return sig_key


class DmxModel(DmxModelMixin):
    @classmethod
    def from_torch(
        cls, model: torch.nn.Module, additional_dmx_aware_mappings=None
    ) -> torch.nn.Module:
        if not isinstance(model, cls):
            _cls = model.__class__
            model.class_for_deserialization = _cls
            model.__class__ = _cls.__class__("Dmx" + _cls.__name__, (_cls, cls), {})
        model.additional_dmx_aware_mappings = additional_dmx_aware_mappings
        model.transformed = False
        model._gm = None
        model._gms = {}
        model._monitoring_records = None
        model._runtime_records = None
        from dmx.compressor import config_rules

        model._dmx_configuration_queue = [(None, config_rules.BASELINE)]

        def temp_forward(_m, *_args, **_kwargs):
            _is_training = _m.training
            sig_key = DmxModel.to_signature_key(_m, _args, _kwargs)
            if not _m.transformed or not DmxModel.is_same_signature(_m, _args, _kwargs):
                warnings.warn("Model transformation triggered")
                _m.tracing_kwargs = (
                    DmxModel.deepcopy_args(_args),
                    DmxModel.deepcopy_args(_kwargs),
                )
                if sig_key in _m._gms:
                    warnings.warn("Reusing graph module from past")
                    _m._gm = _m._gms[sig_key]
                else:
                    # because some args and kwargs are changed in place in forward, we need to keep a copy fo the _args and _kwargs
                    transform_args, transform_kwargs = DmxModel.deepcopy_args(
                        _args
                    ), DmxModel.deepcopy_args(_kwargs)
                    DmxModel._add_transformed_gm(_m, transform_args, transform_kwargs)
                    _m._gms[sig_key] = _m._gm

                _m._forward = DmxModel._get_transformed_forward(_m)

                _m.transformed = True
                # configure the model
                for _config, _rules in _m._dmx_configuration_queue:
                    _m._apply_config(_config, *_rules)
                _m.train(_is_training)
                _m.forward = partial(temp_forward, _m)

                return _m._forward(*_args, **_kwargs)

            return _m._forward(*_args, **_kwargs)

        model.old_forward = model.forward
        model.forward = partial(temp_forward, model)

        return model

    def visualize_graph(self, out_file="graph"):
        if not self.transformed:
            raise RuntimeError(
                "A forward pass is needed before model can be visualized!"
            )
        tracing_args, tracing_kwargs = self.tracing_kwargs
        inputs = tuple(
            signature(self.old_forward)
            .bind(*tracing_args, **tracing_kwargs)
            .arguments.values()
        )
        visualize_graph(self._gm, inputs, out_file)

    def make_compiler_graphs(self):
        if not hasattr(self, "compiler_graphs"):
            self._compiler_graphs = dict()
        for key, gm in self._gms.items():
            if key not in self._compiler_graphs:
                self._compiler_graphs[key] = make_compiler_graph(gm)


class DmxConfig(dict):
    r"""
    This is a dict of Dmx-specific configurations for a DmxModel
    This defines the 'states' to be optimized
    """

    @classmethod
    def from_model(cls, model: torch.nn.Module, freeze=False):
        """
        A function that stores state and ops format of the model in a DmxConfig object

        Args:
            model (Model): Target model for creating the DmxConfig

        Returns:
            A DmxConfig object that stores state and ops format of the model in a DmxConfig object
        """
        return cls({n: m.dmx_config(freeze) for n, m in model.named_dmx_modules()})

    @classmethod
    def from_yaml(cls, fname):
        """
        A function that creates a DmxConfig object from a yaml file

        Args:
            fname (str): file path of the yaml file

        Returns:
            A DmxConfig object vreated from yaml file
        """
        from dmx.compressor.utils.io import load_config_file

        return cls(load_config_file(fname))

    def to_yaml(self, fname):
        """
        A function that stores the content of a DmxConfig object to a yaml file

        Args:
            fname (str): file path of the yaml file
        """
        from dmx.compressor.utils.io import save_config_file

        save_config_file(dict(self), fname)

    @property
    def module_names(self):
        """
        Returns the module names in the DmxConfig object
        """
        return self.keys()


class DmxConfigRule(SimpleNamespace):
    r"""
    This is a rule that specifies how to transform from DmxConfig to DmxConfig
    This defines the 'action' in the state space

    Args:
        module_types (Optional[Tuple]): Types of modules to apply transformation. Defaults to empty Tuple.
        name_re (Optional[str]): String for matching module name patterns. Defaults to empty str.
        module_config (Optional[DmxModuleConfig]): DmxModuleConfig that specifies the ops formats for a module. Defaults to empty DmxModuleConfig.

    Attributes:
        module_types (Tuple): Types of modules to apply transformation.
        name_rule (Pattern): Pattern for matching module names.
        module_config (DmxModuleConfig): DmxModuleConfig that specifies the ops formats for a module.
    """

    def __init__(
        self,
        module_types=(),
        name_re: str = "",
        module_config: DmxModuleConfig = DmxModuleConfig(),
    ) -> None:
        assert all([issubclass(mt, DmxModule) for mt in module_types])
        self.module_types = module_types
        self.name_rule = re.compile(name_re)
        self.module_config = module_config

    def names_in(self, model_or_config: Union[torch.nn.Module, DmxConfig]):
        """
        Creates a list of module names where the modules are in self.module_types and the names match with self.name_rule.

        Args:
            model_or_config (Union[torch.nn.Module, DmxConfig]): torch.nn.Module or DmxConfig to create the name of modules for.

        Returns:
            A list of module names
        """
        config = (
            model_or_config
            if isinstance(model_or_config, DmxConfig)
            else DmxConfig.from_model(model_or_config, freeze=True)
        )
        return [
            n
            for n in config.module_names
            if any(
                issubclass(config[n]["instance_of"], mod_type)
                for mod_type in self.module_types
            )
            and self.name_rule.match(n)
        ]

    def apply_to(self, model_or_config: Union[torch.nn.Module, DmxConfig]):
        """
        A function that sets format of ops according to self.module_config for modules selected by self.module_types and
        self.name_rule on a model or DmxConfig

        Args:
            model_or_config (Union[Model, DmxConfig]): Model or DmxConfig to apply transformation on.
        """
        target_module_names = self.names_in(model_or_config)
        if isinstance(model_or_config, torch.nn.Module):
            for n, m in model_or_config.named_dmx_modules():
                if n in target_module_names and isinstance(m, self.module_types):
                    m.configure(self.module_config)
        else:
            config = model_or_config
            for n in target_module_names and any(
                issubclass(config[n]["instance_of"], mod_type)
                for mod_type in self.module_types
            ):
                config[n].update(self.module_config)


# alias for backward compatibility, to be deprecated
DmxTransformation = DmxConfigRule


class DmxPipelineMixin:
    def configure(
        self,
        dmx_config_dict: Dict[str, Optional[DmxConfig]],
        dmx_transformation_dict: Optional[Dict[str, Optional[DmxConfigRule]]] = None,
    ) -> None:
        for _n, _m in self.named_dmx_models():
            tr = [] if dmx_transformation_dict is None else dmx_transformation_dict[_n]
            _m.transform(dmx_config_dict[_n], *tr)

    transform = configure  # NOTE: to be deprecated

    @contextmanager
    def counting_flops(self, zero: bool = True) -> None:
        with ExitStack() as stack:
            yield [
                stack.enter_context(_m.counting_flops(zero))
                for _, _m in self.named_dmx_models()
            ]

    def eval(self):
        for _m in self.model_dict.values():
            _m.eval()

    @property
    def dmx_config_dict(self) -> Dict[str, DmxConfig]:
        return {n: m.dmx_config for n, m in self.named_dmx_models()}

    def named_dmx_modules(self):
        return (
            (f"{_model_name}.{_module_name}", _module)
            for _model_name, _model in self.named_dmx_models()
            for _module_name, _module in _model.named_dmx_modules()
        )

    def get_model_by_name(self, model_name: str) -> torch.nn.Module:
        return self.model_dict[model_name]


class DmxSimplePipeline(DmxPipelineMixin):
    def __init__(
        self,
        model_dict: OrderedDict,
        preproc=torch.nn.Identity(),
        postproc=torch.nn.Identity(),
        hf: bool = False,
        concrete_args_dict: Optional[Dict] = None,
    ) -> None:
        self.model_dict = model_dict
        self.preproc = preproc
        self.postproc = postproc
        self.hf = hf
        self.concrete_args_dict = concrete_args_dict

    def __call__(self, *args, **kwargs):
        _out = self.postproc(
            torch.nn.Sequential(*[m for m in self.model_dict.values()])(
                self.preproc(*args, **kwargs)
            )
        )
        return _out

    def named_dmx_models(self):
        r"Returns a generator of named DmxModel instances"
        return ((n, m) for n, m in self.model_dict.items() if isinstance(m, Model))

    def to(self, torch_device: Optional[Union[str, torch.device]] = None):
        if torch_device is not None:
            for _n, _m in self.named_dmx_models():
                _m.to(torch_device)
        return self

    @property
    def op_set(self):
        return set.union(*[get_op_set_from(_m) for _m in self.model_dict.values()])


class Model(DmxSimplePipeline):
    r"""
    This is a backward-compatible placeholder for legacy models.
    It is not recommended to use this container; instead, use DmxSimplePipeline directly.
    TODO: to be deprecated

    """

    def __init__(
        self,
        body,
        head=torch.nn.Identity(),
        tail=torch.nn.Identity(),
        hf: bool = False,
        concrete_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_dict=dict(model=DmxModel.from_torch(body, concrete_args)),
            preproc=head,
            postproc=tail,
            hf=hf,
            concrete_args_dict=dict(model=concrete_args),
        )
        self.body = self.model_dict["model"]
        self.head = self.preproc
        self.tail = self.postproc

    @property
    def op_set(self):
        r"Returns a set of unique ops present in the model"
        return set.union(*[get_op_set_from(_m) for _m in self._gms.values()])
