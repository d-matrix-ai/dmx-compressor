import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from dmx.compressor import format
from dmx.compressor import nn as dmxnn
from dmx.compressor.modeling import DmxModel, DmxConfigRule, DmxModuleConfig
from dmx.compressor.numerical.format import Same
from dmx.compressor.sparse import Dense, Sparseness

RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lenet5(nn.Module):
    r"""
    Taken from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # Max pooling over a (2, 2) window
        self.mp1 = nn.MaxPool2d((2, 2))
        # If the size is a square, you can specify with a single number
        self.mp2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


conv_quantize_input = DmxConfigRule(
    module_types=(dmxnn.Conv2d,),
    module_config=DmxModuleConfig(input_format=format.BFP12_128),
)
conv_quantize_weight = DmxConfigRule(
    module_types=(dmxnn.Conv2d,),
    module_config=DmxModuleConfig(weight_format=format.BFP12_128),
)
conv_sparsify_weight = DmxConfigRule(
    module_types=(dmxnn.Conv2d,),
    module_config=DmxModuleConfig(
        weight_sparseness=Sparseness.from_shorthand("TOPK{0.5}(U)")
    ),
)
fc_quantize_input = DmxConfigRule(
    module_types=(dmxnn.Linear,),
    module_config=DmxModuleConfig(input_format=format.BFP12_128),
)
fc_quantize_weight = DmxConfigRule(
    module_types=(dmxnn.Linear,),
    module_config=DmxModuleConfig(weight_format=format.BFP12_128),
)
fc_sparsify_weight = DmxConfigRule(
    module_types=(dmxnn.Linear,),
    module_config=DmxModuleConfig(
        weight_sparseness=Sparseness.from_shorthand("TOPK{0.5}(U)")
    ),
)


def _create_input(batch_size: int = 1):
    return torch.randn(batch_size, 1, 32, 32)


def _create_model(transformations):
    _model = DmxModel.from_torch(Lenet5())
    conv_quantize_input.apply_to(_model)
    fc_quantize_input.apply_to(_model)
    for tr in transformations:
        tr.apply_to(_model)
    with _model.calibrating_smoothquant(), torch.no_grad():
        _model(_create_input(batch_size=128))
    _model.eval()
    return _model


def _check_if_folded(model: DmxModel) -> bool:
    return all(
        [
            all(
                [
                    m.bias_cast is None or isinstance(m.bias_cast.format, Same),
                    m.weight_sparsifier is None
                    or isinstance(m.weight_sparsifier.sparseness, Dense),
                    m.smoothquant is None or m.smoothquant.fused_to_weight[0] == 1,
                    m.weight_cast is None or isinstance(m.weight_cast.format, Same),
                ]
            )
            for _, m in model.named_dmx_modules()
        ]
    )


@pytest.mark.parametrize(
    "dmx_transformations",
    (
        (),
        (conv_quantize_weight,),
        (fc_quantize_weight,),
        (conv_sparsify_weight,),
        (fc_sparsify_weight,),
        (conv_quantize_weight, fc_quantize_weight),
        (conv_sparsify_weight, fc_sparsify_weight),
        (conv_quantize_weight, conv_sparsify_weight),
        (fc_quantize_weight, fc_sparsify_weight),
        (
            conv_quantize_weight,
            fc_quantize_weight,
            conv_sparsify_weight,
            fc_sparsify_weight,
        ),
    ),
)
def test_fold_weights_and_biases(dmx_transformations):
    x = _create_input()
    model = _create_model(dmx_transformations)
    with torch.no_grad():
        y_ref = model(x)
    model.fold_weights_and_biases()
    assert _check_if_folded(model)
    with torch.no_grad():
        y = model(x)
    assert torch.all(y == y_ref)
