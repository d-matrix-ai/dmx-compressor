from __future__ import print_function
import argparse
import os
from re import M
from dotenv import load_dotenv
import ctypes
import ipdb
import numpy.ctypeslib as ctl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import transformers
from data import MNIST

# env
load_dotenv(verbose=True)
DATA_PATH = os.getenv("DATA_PATH") or "./data_dir"
MODEL_PATH = os.getenv("MODEL_PATH") or "./model_dir"

# components
libc = ctypes.CDLL("./src/modules/libc/actfunction.so")
libc.gelu.argtypes = [ctl.ndpointer(np.float64, flags="aligned, c_contiguous")]
libc.gelu.restype = ctl.ndpointer(
    np.float64, shape=(1000 * 1024,), flags="aligned, c_contiguous"
)


def dmgelu(arr, lid):
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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return dmgelu(x, None)


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
    parser = argparse.ArgumentParser(description="MNIST LeNet quantization explorations")
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
    parser.add_argument(
        "--act-fn",
        type=str,
        default="relu",
        help="activation function (default: 'relu')",
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
    
    model_dir = os.path.join(MODEL_PATH, "mnist-lenet"+f"-{args.width}"*args.depth+f"-{args.act_fn}")
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    nn.GELU = DMGELU
    from models import LeNet
    model = LeNet(hidden_dims=[args.width,]*args.depth, act_func=args.act_fn).to(device)

    # import ipdb; ipdb.set_trace()
    if os.path.exists(os.path.join(model_dir, "trained.pt")) and not args.retrain_model:
        model.load_state_dict(torch.load(os.path.join(model_dir, "trained.pt")))
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, nesterov=True
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
        # optimizer = transformers.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=0, num_training_steps=args.epochs
        # )

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, ds.train, optimizer, epoch)
            scheduler.step()

        torch.save(model.state_dict(), os.path.join(model_dir, "trained.pt"))

    # quantized_model = torch.quantization.quantize_dynamic(
    #     model, {torch.nn.Linear}, dtype=torch.qint8
    # )

    test(model, device, ds.test)
    # test(quantized_model, device, ds.test)

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
