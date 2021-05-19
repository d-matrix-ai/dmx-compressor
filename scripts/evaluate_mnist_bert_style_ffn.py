from __future__ import print_function
import argparse
import os
from re import M
from dotenv import load_dotenv
import ctypes
import ipdb
import numpy.ctypeslib as ctl
import numpy as np
from qtorch.quant.quant_function import float_quantize
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import transformers
import qtorch
from qtorch.quant import Quantizer, fixed_point_quantize, block_quantize, float_quantize
from data import MNIST
from transformers.utils.dummy_pt_objects import BertForMaskedLM
from functools import partial

# env
load_dotenv(verbose=True)
DATA_PATH = os.getenv("DATA_PATH") or "./data_dir"
MODEL_PATH = os.getenv("MODEL_PATH") or "./model_dir"

# numerical
IMC_INPUT_FORMAT = qtorch.BlockFloatingPoint(wl=23, dim=1)
IMC_WEIGHT_FORMAT = qtorch.BlockFloatingPoint(wl=23, dim=1)
IMC_BIAS_FORMAT = qtorch.FloatingPoint(exp=8, man=23)
IMC_OUTPUT_FORMAT = qtorch.FloatingPoint(exp=8, man=23)

SIMD_INPUT_FORMAT = qtorch.FixedPoint(wl=24, fl=4)
SIMD_OUTPUT_FORMAT = qtorch.FixedPoint(wl=24, fl=4)

# components
libc = ctypes.CDLL("./src/modules/libc/actfunction.so")
libc.gelu.argtypes = [ctl.ndpointer(np.float64, flags="aligned, c_contiguous")]
libc.gelu.restype = ctl.ndpointer(
    np.float64, shape=(1000 * 1024,), flags="aligned, c_contiguous"
)


def dmgelu(arr, lid=None):
    mx = torch.max(arr).item()
    inp = torch.reshape(arr, (1000, 1024))
    inp = inp.numpy().astype(np.float64, order="C")
    # print(mx)
    # inp *= (8/mx)
    out = libc.gelu(inp)
    out = torch.Tensor(out)
    # out *= (mx/8)
    # out /= 32
    out = torch.reshape(out, (1000, 1024))
    return out


class DMGELU(nn.Module):
    def __init__(self, bit_scale=0, use_lut=False):
        super().__init__()
        self.qinput = Quantizer(
            forward_number=SIMD_INPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.qoutput = Quantizer(
            forward_number=SIMD_OUTPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.bit_scale = bit_scale
        self.use_lut = use_lut

    def forward(self, x):
        # x = self.qinput(x)
        if self.bit_scale is not None:
            x = fixed_point_quantize(
                x, wl=24, fl=self.bit_scale, symmetric=True, rounding="nearest"
            )
        x = dmgelu(x) if self.use_lut else F.gelu(x)
        if self.bit_scale is not None:
            x = fixed_point_quantize(
                x, wl=24, fl=self.bit_scale, symmetric=True, rounding="nearest"
            )
        # x = self.qoutput(x)
        return x


class DMReLU(nn.Module):
    def __init__(self, bit_scale=0, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.qinput = Quantizer(
            forward_number=SIMD_INPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.qoutput = Quantizer(
            forward_number=SIMD_OUTPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.bit_scale = bit_scale

    def forward(self, x):
        if self.bit_scale is not None:
            # x = self.qinput(x)
            x = fixed_point_quantize(
                x, wl=24, fl=self.bit_scale, symmetric=True, rounding="nearest"
            )
        x = torch.relu_(x) if self.inplace else torch.relu(x)
        if self.bit_scale is not None:
            x = fixed_point_quantize(
                x, wl=24, fl=self.bit_scale, symmetric=True, rounding="nearest"
            )
            # x = self.qoutput(x)
        return x


class DMLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prec: int = 8,
        block_size: int = 64,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.qweight = Quantizer(
            forward_number=IMC_WEIGHT_FORMAT,
            forward_rounding="nearest",
        )
        self.qbias = Quantizer(
            forward_number=IMC_BIAS_FORMAT,
            forward_rounding="nearest",
        )
        self.qinput = Quantizer(
            forward_number=IMC_INPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.qoutput = Quantizer(
            forward_number=IMC_OUTPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.prec = prec
        self.block_size = block_size

    def forward(self, input: Tensor) -> Tensor:

        _weights = torch.split(self.weight, self.block_size, dim=1)
        _weight = torch.cat(
            [
                block_quantize(_w.clone(), wl=self.prec, dim=0, rounding="nearest")
                for _w in _weights
            ],
            dim=1,
        )
        # _weight = self.qweight(self.weight)
        # _weight = block_quantize(self.weight, wl=self.prec, dim=0, rounding="nearest")

        # _bias = self.qbias(self.bias) if self.bias is not None else None
        _bias = self.bias  # (
        #     float_quantize(self.bias, exp=8, man=23) if self.bias is not None else None
        # )

        _inputs = torch.split(input, self.block_size, dim=1)
        _input = torch.cat(
            [
                block_quantize(_i.clone(), wl=self.prec, dim=0, rounding="nearest")
                for _i in _inputs
            ],
            dim=1,
        )
        # _input = self.qinput(input)
        # _input = block_quantize(input, wl=self.prec, dim=0, rounding="nearest")

        _output = F.linear(_input, _weight, _bias)

        # output = self.qoutput(_output)
        output = _output  # float_quantize(_output, exp=8, man=23)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="MNIST BERT-style FFN quantization explorations"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        metavar="N",
        help="number of blocks (default: 2)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        metavar="N",
        help="hidden dimension (default: 1024)",
    )
    # parser.add_argument(
    #     "--act-fn",
    #     type=str,
    #     default="relu",
    #     help="activation function (default: 'relu')",
    # )
    parser.add_argument(
        "--use-lut",
        action="store_true",
        default=False,
        help="use LUT for act_fn (default: False)",
    )
    parser.add_argument(
        "--bfp",
        type=int,
        default=None,
        help="BFP precision, 12-16, None for FP (default: None)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="BFP block size (default: 1024)",
    )
    parser.add_argument(
        "--xp",
        type=int,
        default=None,
        help="XP bias, None for FP (default: None)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--retrain-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    ds = MNIST(
        data_dir=os.path.join(DATA_PATH, "mnist"),
        cuda=use_cuda,
        train_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )

    model_dir = os.path.join(
        MODEL_PATH, "mnist-bert_style_ffn" + f"-{args.width}" * args.depth
    )
    print(f"Model directory: {model_dir}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if args.xp is not None:
        nn.ReLU = partial(DMReLU, bit_scale=args.xp)
    nn.GELU = partial(DMGELU, bit_scale=args.xp, use_lut=args.use_lut)

    if args.bfp is not None:
        nn.Linear = partial(DMLinear, prec=args.bfp - 8, block_size=args.block_size)

    cf_str = ''
    if args.use_lut:
        cf_str += "gelu-lut"
    cf_str += f"-bfp{args.bfp}" if args.bfp is not None else "-fp32"
    cf_str += f"-xp{args.xp}" if args.xp is not None else "-fp32"
    print(f"Configuration: {cf_str}")

    from models import BERTStyleFFN

    model = BERTStyleFFN(depth=args.depth, width=args.width).to(device)

    if os.path.exists(os.path.join(model_dir, "trained.pt")) and not args.retrain_model:
        model.load_state_dict(torch.load(os.path.join(model_dir, "trained.pt")))
    else:
        optimizer = transformers.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=args.epochs
        )

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, ds.train, optimizer, epoch)
            scheduler.step()

        torch.save(model.state_dict(), os.path.join(model_dir, "trained.pt"))

    test(model, device, ds.test)

    # import ipdb; ipdb.set_trace()

    # from apex.contrib.sparsity import ASP

    # ASP.prune_trained_model(model.body, optimizer)
    # test(model, device, ds.test)

    # optimizer = transformers.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    # for epoch in range(1, args.fine_tune_epochs+1):
    #     train(args, model, device, ds.train, optimizer, epoch)
    # test(model, device, ds.test)

    # if args.save_model:
    #     torch.save(model.state_dict(), "model_sparsified.pt")


if __name__ == "__main__":
    main()
