import pytest
import torch
from typing import Tuple, Optional, Union
from mltools import dmx, sparse, numerical
from dataclasses import dataclass

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)


@dataclass(frozen=True)
class TestCase:
    # Given:
    layer_type: dmx.nn.DmxModule
    layer_dims: Tuple[
        int, int, Optional[Tuple[int, ...]], Optional[int], Optional[int], Optional[str]
    ]
    input_dims: Tuple[int, Optional[Tuple[int, ...]]]
    weight_sparseness: sparse.Sparseness
    input_format: Union[numerical.Format, torch.dtype]
    weight_format: Union[numerical.Format, torch.dtype]
    # Ensure:
    _weight_elem_count: float
    _weight_size_in_bytes: float
    _flops: float
    _bops: float

    def __str__(self):
        return f"{self.layer_type.__name__}-{self.layer_dims}-{self.input_dims}-{repr(self.weight_sparseness)}-{repr(self.input_format)}-{repr(self.weight_format)}"


test_cases = [
    TestCase(
        layer_type=dmx.nn.Linear,
        layer_dims=(16, 16, None, None, None, None),
        input_dims=(1, None),
        weight_sparseness=sparse.Dense(),
        input_format=torch.float32,
        weight_format=torch.float32,
        # --------------------------------
        _weight_elem_count=256.0,
        _weight_size_in_bytes=1024.0,
        _flops=256.0,
        _bops=262144.0,
    ),
    TestCase(
        layer_type=dmx.nn.Linear,
        layer_dims=(8, 24, None, None, None, None),
        input_dims=(8, None),
        weight_sparseness=dmx.sparseness.BTK8_4_LD,
        input_format=dmx.format.BFP16_128,
        weight_format=dmx.format.BFP12_128,
        # --------------------------------
        _weight_elem_count=96.0,
        _weight_size_in_bytes=48.75,
        _flops=768.0,
        _bops=25155.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv2d,
        layer_dims=(8, 32, (3, 3), 1, 1, "same"),
        input_dims=(8, (10, 10)),
        weight_sparseness=sparse.Dense(),
        input_format=torch.float16,
        weight_format=torch.float16,
        # --------------------------------
        _weight_elem_count=2304.0,
        _weight_size_in_bytes=4608.0,
        _flops=1843200.0,
        _bops=471859200.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv2d,
        layer_dims=(8, 32, (3, 3), 1, 1, "valid"),
        input_dims=(8, (10, 10)),
        weight_sparseness=sparse.Dense(),
        input_format=torch.float16,
        weight_format=torch.float16,
        # --------------------------------
        _weight_elem_count=2304.0,
        _weight_size_in_bytes=4608.0,
        _flops=1179648.0,
        _bops=301989888.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv2d,
        layer_dims=(8, 32, (3, 3), 1, 1, "same"),
        input_dims=(8, (10, 10)),
        weight_sparseness=dmx.sparseness.BTK8_4_FD,
        input_format=torch.float16,
        weight_format=torch.float16,
        # --------------------------------
        _weight_elem_count=1152.0,
        _weight_size_in_bytes=2304.0,
        _flops=921600.0,
        _bops=235929600.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv2d,
        layer_dims=(8, 32, (5, 5), 1, 1, "valid"),
        input_dims=(8, (10, 10)),
        weight_sparseness=dmx.sparseness.BTK8_2_FD,
        input_format=dmx.format.INT8,
        weight_format=dmx.format.INT8,
        # --------------------------------
        _weight_elem_count=1600.0,
        _weight_size_in_bytes=1600.0,
        _flops=460800.0,
        _bops=29491200.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv2d,
        layer_dims=(8, 32, (7, 7), 1, 1, "same"),
        input_dims=(16, (13, 13)),
        weight_sparseness=dmx.sparseness.BTK8_2_FD,
        input_format=dmx.format.INT8,
        weight_format=dmx.format.INT4,
        # --------------------------------
        _weight_elem_count=3136.0,
        _weight_size_in_bytes=1568.0,
        _flops=8479744.0,
        _bops=271351808.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv2d,
        layer_dims=(8, 32, (7, 7), 1, 1, "valid"),
        input_dims=(16, (13, 13)),
        weight_sparseness=dmx.sparseness.BTK8_2_FD,
        input_format=dmx.format.INT8,
        weight_format=dmx.format.INT4,
        # --------------------------------
        _weight_elem_count=3136.0,
        _weight_size_in_bytes=1568.0,
        _flops=2458624.0,
        _bops=78675968.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv2d,
        layer_dims=(8, 32, (7, 7), 1, 4, "same"),
        input_dims=(8, (13, 13)),
        weight_sparseness=sparse.Dense(),
        input_format=dmx.format.INT4,
        weight_format=dmx.format.INT4,
        # --------------------------------
        _weight_elem_count=3136.0,
        _weight_size_in_bytes=1568.0,
        _flops=4239872.0,
        _bops=67837952.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv2d,
        layer_dims=(32, 32, (3, 3), 2, 4, "valid"),
        input_dims=(8, (10, 10)),
        weight_sparseness=dmx.sparseness.BTK8_4_FD,
        input_format=dmx.format.BFP14_128,
        weight_format=dmx.format.BFP12_128,
        # --------------------------------
        _weight_elem_count=1152.0,
        _weight_size_in_bytes=585.0,
        _flops=147456.0,
        _bops=3631680.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv1d,
        layer_dims=(8, 16, (3,), 1, 2, "same"),
        input_dims=(16, (10,)),
        weight_sparseness=sparse.Dense(),
        input_format=torch.float16,
        weight_format=torch.float16,
        # --------------------------------
        _weight_elem_count=192.0,
        _weight_size_in_bytes=384.0,
        _flops=30720.0,
        _bops=7864320.0,
    ),
    TestCase(
        layer_type=dmx.nn.Conv1d,
        layer_dims=(8, 16, (3,), 2, 2, "valid"),
        input_dims=(16, (10,)),
        weight_sparseness=sparse.Dense(),
        input_format=dmx.format.INT8,
        weight_format=dmx.format.INT8,
        # --------------------------------
        _weight_elem_count=192.0,
        _weight_size_in_bytes=192.0,
        _flops=12288.0,
        _bops=786432.0,
    ),
]


def _create_module(cls, layer_dims, weight_sparseness, input_format, weight_format):
    ch_in, ch_out, ker_size, stride, groups, padding = layer_dims
    if cls == dmx.nn.Linear:
        _module = cls(
            in_features=ch_in,
            out_features=ch_out,
        )
    elif cls in (dmx.nn.Conv1d, dmx.nn.Conv2d):
        _module = cls(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=ker_size,
            stride=stride,
            groups=groups,
            padding=padding,
        )
    else:
        raise ValueError("unsupported module class {cls}")
    _module.weight_sparsifier.configure(sparseness=weight_sparseness)
    if isinstance(input_format, numerical.Format):
        _module.input_casts.input_cast.set_format(input_format)
    if isinstance(weight_format, numerical.Format):
        _module.weight_cast.set_format(weight_format)
    elif weight_format == torch.float16:  # in order to work on both CPU/GPU
        _module.weight_cast.set_format(dmx.format.FLOAT16)
    return _module


def _create_test_input(batch_size, in_channel, img_size, input_format):
    shape = (batch_size, in_channel)
    if img_size is not None:
        shape += img_size
    dtype = input_format if isinstance(input_format, torch.dtype) else torch.float32
    return torch.randn(torch.Size(shape)).to(dtype)


@pytest.fixture(
    scope="class",
    params=test_cases,
    ids=[str(tc) for tc in test_cases],
)
def _test_run(request):
    p = request.param
    ch_in, ch_out, ker_size, stride, groups, padding = p.layer_dims
    batch_size, img_size = p.input_dims
    module = _create_module(
        p.layer_type,
        p.layer_dims,
        p.weight_sparseness,
        p.input_format,
        p.weight_format,
    )
    x_ref = _create_test_input(
        batch_size=batch_size,
        in_channel=ch_in,
        img_size=img_size,
        input_format=p.input_format,
    )
    with module.counting_flops(zero=True), torch.no_grad():
        module(x_ref)

    yield module, p


@pytest.mark.usefixtures("_test_run")
def test_perf_proxy(_test_run):
    module, p = _test_run
    assert module.weight_elem_count == p._weight_elem_count
    assert module.weight_size_in_bytes == p._weight_size_in_bytes
    assert module.flops == p._flops
    assert module.bops == p._bops
