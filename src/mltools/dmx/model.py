import torch
import re
from collections import deque, OrderedDict
from inspect import signature, _empty
from types import SimpleNamespace
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Optional, Union, Sequence, get_args, get_origin
from functools import partial
from mltools import dmx
from mltools.dmx.nn import *
from mltools.fx.transform import substitute_transform
from mltools.fx.transformer import get_op_set_from
import functools


class DmxModelMixin:
    transformed: bool = False
    _dmx_configurations_to_be_applied: deque = (
        deque()
    )  # stores (config, rules) to be applied

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
        if not self.transformed:
            self._dmx_configurations_to_be_applied.append((config, rules))
        else:
            if config is not None:
                if isinstance(config, str):
                    config = DmxConfig.from_yaml(config)

                for n, m in self.named_dmx_modules():
                    if n in config:
                        m.configure(config[n])

            for _r in rules:
                _r.apply_to(self)

        return self

    transform = configure  # NOTE: to be deprecated

    @property
    def op_set(self):
        r"Returns a set of unique ops present in the model"
        return get_op_set_from(self._gm)

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
        return ((n, m) for n, m in self.named_modules() if is_configurable(m))

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
        from mltools.utils import print_model_tree

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

    @contextmanager
    def keep_dmx_config(self):
        _dmx_config = self.dmx_config
        yield self
        self.configure(_dmx_config)

    @contextmanager
    def counting_flops(self, zero: bool = True):
        with ExitStack() as stack:
            yield [
                stack.enter_context(m.counting_flops(zero))
                for _, m in self.named_dmx_modules()
            ]

    @staticmethod
    def _save_specific_layers_state_dict_and_register_urls(
        specific_layers: Dict[str, Sequence[DmxModule]],
        save_checkpoint_to: Optional[str],
    ):
        if save_checkpoint_to is not None:
            for _, m in specific_layers:
                m.save_state_dict_and_register_url(parent_dir=save_checkpoint_to)

    @contextmanager
    def calibrating_weights(
        self,
        specific_layers: Optional[Dict[str, Sequence[DmxModule]]] = None,
        save_checkpoint_to: Optional[str] = None,
        **hyperparams,
    ):
        if specific_layers is None:
            specific_layers = self.named_dmx_modules()
        for _, _m in specific_layers:
            _m.set_weight_calibrator(**hyperparams)
        with torch.no_grad(), ExitStack() as stack:
            yield [
                stack.enter_context(m.calibrating_weight()) for _, m in specific_layers
            ]
        self._save_specific_layers_state_dict_and_register_urls(
            specific_layers, save_checkpoint_to
        )

    @contextmanager
    def calibrating_activations(
        self,
        specific_layers: Optional[Dict[str, Sequence[DmxModule]]] = None,
        save_checkpoint_to: Optional[str] = None,
        **hyperparams,
    ):
        if specific_layers is None:
            specific_layers = self.named_dmx_modules()
        for _, _m in specific_layers:
            _m.set_activation_calibrator(**hyperparams)
        with torch.no_grad(), ExitStack() as stack:
            yield [
                stack.enter_context(m.calibrating_activation())
                for _, m in specific_layers
            ]
        self._save_specific_layers_state_dict_and_register_urls(
            specific_layers, save_checkpoint_to
        )

    @contextmanager
    def calibrating_smoothquant(
        self,
        specific_layers: Optional[Dict[str, Sequence[DmxModule]]] = None,
        save_checkpoint_to: Optional[str] = None,
        **hyperparams,
    ):
        if specific_layers is None:
            specific_layers = self.named_dmx_modules()
        for _, _m in specific_layers:
            _m.set_smoothquant_params(**hyperparams)
        with torch.no_grad(), ExitStack() as stack:
            yield [
                stack.enter_context(m.calibrating_smoothquant())
                for _, m in specific_layers
            ]
        self._save_specific_layers_state_dict_and_register_urls(
            specific_layers, save_checkpoint_to
        )

    @contextmanager
    def optimal_brain_compressing(
        self,
        specific_layers: Optional[Dict[str, Sequence[DmxModule]]] = None,
        save_checkpoint_to: Optional[str] = None,
        **hyperparams,
    ):
        if specific_layers is None:
            specific_layers = self.named_dmx_modules()
        with torch.no_grad(), ExitStack() as stack:
            yield [
                stack.enter_context(m.optimal_brain_compressing(**hyperparams))
                for _, m in specific_layers
            ]
        self._save_specific_layers_state_dict_and_register_urls(
            specific_layers, save_checkpoint_to
        )


class DmxModel(torch.nn.Module):
    @staticmethod
    def _get_transformed_forward(_model, args, kwargs):
        print("Model transformation triggered")
        _model.forward = _model.old_forward
        _mod_signature = signature(_model.forward)
        _output_cls = _mod_signature.return_annotation
        if get_origin(_output_cls) is Union:  # NOTE: this is error-prone
            _output_cls = get_args(_output_cls)[1]
            assert issubclass(
                _output_cls, transformers.modeling_utils.ModelOutput
            )  # NOTE: using this to guard against abuse
        elif _output_cls is _empty:
            _output_cls = None

        if "cache_position" in kwargs:
            kwargs["cache_position"] = None
        # remove kwargs with value None
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # boolean inputs will affect tracing and need to be set as concrete args
        bool_inputs = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
        kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

        input_names = signature(_model.forward).bind(*args, **kwargs).arguments.keys()
        dummy_inputs = {}
        for k in input_names:
            if k not in kwargs:
                dummy_inputs[k] = args[0]
            else:
                dummy_inputs[k] = kwargs[k]
        _model._gm = substitute_transform(
            _model,
            hf=_model.hf,
            input_names=input_names,
            concrete_args=bool_inputs,
            dummy_inputs=dummy_inputs,
        )
        # some inputs were removed from input names due to None or bool, we want to add it back to maintain original input signature
        intersection = set(_model.tracing_kwargs.keys()) - set(kwargs.keys())
        if intersection:
            # find first none placehoder node
            node_list = _model._gm.graph.nodes
            for node in node_list:
                if node.op != "placeholder":
                    break
            for inp in intersection:
                with _model._gm.graph.inserting_before(node):
                    _model._gm.graph.placeholder(inp)
            _model._gm.recompile()
        _model._output_cls = _output_cls
        _forward = (
            (
                lambda *_args, **_kwargs: _output_cls(
                    **(_model._gm.forward(*_args, **_kwargs))
                )
            )
            if _model._output_cls is not None
            else _model._gm.forward
        )
        return _forward

    @staticmethod
    def is_same_signature(_model, kwargs):
        tracing_kwargs = _model.tracing_kwargs
        # if kwargs has different keys, need to retrace
        if tracing_kwargs.keys() != kwargs.keys():
            return False
        # comparing kwargs values between tracing kwarg and new kwargs
        for k in kwargs.keys():
            # if bool argument has different values, need to retrace
            if isinstance(kwargs[k], bool) and kwargs[k] != tracing_kwargs[k]:
                return False
            # if one in None and other is not none, need to retrace
            if not isinstance(kwargs[k], bool) and (kwargs[k] is None) != (
                tracing_kwargs[k] is None
            ):
                return False
        return True

    @classmethod
    def from_torch(
        cls,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        if DmxModelMixin not in model.__class__.__bases__:
            model.__class__.__bases__ += (DmxModelMixin,)
            model._gm = None
            model.transformed = False
            model.hf = model.__class__.__module__.startswith("transformers")

            def temp_forward(_m, *_args, **_kwargs):
                _is_training = _m.training
                if not _m.transformed or not DmxModel.is_same_signature(_m, _kwargs):

                    if _m.transformed:
                        curr_cfg = _m.dmx_config
                    print("triggering transform")
                    _m.tracing_kwargs = _kwargs.copy()
                    # _kwargs["cache_position"] = None
                    _m._forward = DmxModel._get_transformed_forward(_m, _args, _kwargs)
                    if _m.transformed:
                        _m.configure(curr_cfg)
                    else:
                        _m.transformed = True
                        _m.baseline_config = _m.dmx_config  # BASELINE config recorded
                        while len(_m._dmx_configurations_to_be_applied) != 0:
                            _config, _rules = (
                                _m._dmx_configurations_to_be_applied.popleft()
                            )
                            _m.configure(_config, *_rules)
                    _m.train(_is_training)
                    _m.forward = partial(temp_forward, _m)
                    return _m._forward(*_args, **_kwargs)

                return _m._forward(*_args, **_kwargs)

            model.old_forward = model.forward
            model.forward = partial(temp_forward, model)

        return model


class DmxConfig(dict):
    r"""
    This is a dict of Dmx-specific configurations for a dmx.Model
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
        from mltools.utils import load_config_file

        return cls(load_config_file(fname))

    def to_yaml(self, fname):
        """
        A function that stores the content of a DmxConfig object to a yaml file

        Args:
            fname (str): file path of the yaml file
        """
        from mltools.utils import save_config_file

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
            if config[n]["instance"] in self.module_types and self.name_rule.match(n)
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
                if n in target_module_names and type(m) in self.module_types:
                    m.configure(self.module_config)
        else:
            config = model_or_config
            for n in (
                target_module_names
                and getattr(dmx.nn, config[n]["instance"]) in self.module_types
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
        return ((n, m) for n, m in self.model_dict.items() if isinstance(m, dmx.Model))

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
        return get_op_set_from(self.body._gm)
