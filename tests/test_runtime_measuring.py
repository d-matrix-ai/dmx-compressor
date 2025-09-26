import pytest
import torch
from dmx.compressor.modeling import DmxModel


class LinearRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(50, 50)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        return self.act(self.lin(x))


class NW(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.mods = torch.nn.ModuleList([LinearRelu() for _ in range(n_layers)])

    def forward(self, x):
        for m in self.mods:
            x = m(x)
        return x


@pytest.mark.parametrize("dev_name", ("cpu", "cuda"))
@pytest.mark.parametrize("n_layers", (1, 3, 5))
def test_runtime_measuring(dev_name, n_layers):
    if dev_name == "cuda" and not torch.cuda.is_available():
        # test again on cpu
        dev_name = "cpu"
    device = torch.device(dev_name)
    model = NW(n_layers).to(device)

    model = DmxModel.from_torch(model)
    model.to_baseline_mode()
    model(torch.rand(12, 50).to(device))
    mod_names = [".".join(x[0].split(".")[1:]) for x in list(model.named_dmx_modules())]

    with model.measure_runtimes(device, mod_names):
        model(torch.rand(12, 50).to(device))
    recs = model.get_runtime_records()
    assert all(
        x in recs for x in mod_names
    ), "failed to measure the runtime of some modules in baseline mode"

    model.to_basic_mode()
    with model.measure_runtimes(device, mod_names):
        model(torch.rand(12, 50).to(device))

    recs = model.get_runtime_records()
    assert all(
        x in recs for x in mod_names
    ), "failed to measure the runtime of some modules in baseline mode"
