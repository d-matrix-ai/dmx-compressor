import pytest
import torch
from dmx.compressor.modeling import DmxModel


class LinearRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(50, 50)
        self.act = torch.nn.ReLU()

    def forward(self, x, activations=False):
        return self.act(self.lin(x)) if activations else self.lin(x)


class NW(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.mods = torch.nn.ModuleList([LinearRelu() for _ in range(n_layers)])

    def forward(self, x, activations=False):
        for m in self.mods:
            x = m(x, activations)
        return x


@pytest.mark.parametrize("dev_name", ("cpu", "cuda"))
@pytest.mark.parametrize("n_layers", (1, 3, 5))
@pytest.mark.parametrize("config", ("baseline", "basic"))
def test_monitoring_across_gms(dev_name, n_layers, config):
    if dev_name == "cuda" and not torch.cuda.is_available():
        dev_name = "cpu"
    device = torch.device(dev_name)

    model = NW(n_layers).to(device)
    model = DmxModel.from_torch(model)
    model.to_baseline_mode() if config == "baseline" else model.to_basic_mode()

    # Forward pass with latest _gm missing some submodules
    model(torch.rand(12, 50).to(device), activations=True)
    model(torch.rand(12, 50).to(device), activations=False)

    mod_names = [f'mods.{i}.lin' for i in range(n_layers)] + [f'mods.{i}.act' for i in range(n_layers)]

    with model.monitoring(mod_names):
        model(torch.rand(12, 50).to(device), activations=True)
        model(torch.rand(12, 50).to(device), activations=False)

    recs = model.get_monitoring_records()

    assert all(
        x in recs for x in mod_names
    ), f"monitoring failed to monitor some submodules in {config} mode"
