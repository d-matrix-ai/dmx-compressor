""" Demo: multi-layer perceptron trained on MNIST for Corsair deployment."""
import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST LeNet deployment on Corsair")
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        metavar="N",
        help="number of intermediate blocks (default: 2)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        metavar="N",
        help="hidden dimension (default: 512)",
    )
    parser.add_argument(
        "--train-batch-size",
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
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        metavar="LR",
        help="learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=0.02,
        metavar="LR",
        help="learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
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
    return parser.parse_args()


# env boiler plate
load_dotenv(verbose=True)
DATA_PATH = os.getenv("DATA_PATH") or "./data_dir"
MODEL_PATH = os.getenv("MODEL_PATH") or "./model_dir"
args = parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
pb_wrap = lambda it: tqdm(it, leave=False, dynamic_ncols=True)
args.model_dir = os.path.join(
    MODEL_PATH, "mnist-lenet" + f"-{args.width}" * args.depth
)
writer = SummaryWriter(args.model_dir+'/logs')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    with pb_wrap(train_loader) as loader:
        loader.set_description(f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(loader):
            data, target =  torch.flatten(data, start_dim=1, end_dim=-1).to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()
            optimizer.step()
            loader.set_postfix(
                    loss="\33[91m{:6.4f}\033[0m".format(loss))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        with pb_wrap(test_loader) as loader:
            loader.set_description(f"Evaluation")
            for data, target in loader:
                data, target = torch.flatten(data, start_dim=1, end_dim=-1).to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(
                    F.log_softmax(output, dim=1), target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\tAverage loss: {:.4f} \n\tAccuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def load_data():
    return MNIST(
        data_dir=os.path.join(DATA_PATH, "mnist"),
        cuda=use_cuda,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
    )


def instantiate_model():
    print(f"Model directory: {args.model_dir}")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    return LeNet(
        hidden_dims=[
            args.width,
        ]
        * args.depth,
    ).to(device)


def train_or_load_pretrained(model, ds):
    ckpt = os.path.join(args.model_dir, "trained.pt")
    if os.path.exists(ckpt) and not args.retrain_model:
        print(f"Found pretrained model {ckpt}, loading...")
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    else:
        print("No pretrained model found, training...")
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, nesterov=True
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, ds, optimizer, epoch)
            scheduler.step()

        print(f"Saving model to {ckpt}...")
        torch.save(model.state_dict(), ckpt)
    return model


def fine_tune(model, ds):
    ckpt = os.path.join(args.model_dir, "finetuned.pt")
    if os.path.exists(ckpt) and not args.retrain_model:
        print(f"Found fine-tuned model {ckpt}, loading...")
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    else:
        print("No fine-tuned model found, fine-tuning...")
        optimizer = optim.SGD(model.parameters(), lr=args.fine_tune_lr, momentum=0.9, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.fine_tune_epochs
        )

        for epoch in range(1, args.fine_tune_epochs + 1):
            train(args, model, device, ds, optimizer, epoch)
            scheduler.step()

        print(f"Saving model to {ckpt}...")
        torch.save(model.state_dict(), ckpt)
    return model


def evaluate(model, ds):
    test(model, device, ds)


def dump_onnx(model):
    onnx_file = os.path.join(args.model_dir, "trained.onnx")
    torch.onnx.export(
        model,
        torch.randn(
            (
                1,
                28,
                28,
            ),
            requires_grad=True,
        ).to(device),
        onnx_file,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Dumped to {onnx_file}")


if __name__ == "__main__":

    from mltools import corsair
    corsair.aware()  # pytorch is now extended to be "corsair-aware"

    from mltools.data import MNIST  # mnist data
    from mltools.models import LeNet  # model implementation in pytorch

    dataset = load_data()  # load dataset

    model = corsair.Model(instantiate_model())  # instantiate pytorch model

    model = train_or_load_pretrained(model, dataset.train) # train or load model params

    print(f"\nEvaluation of the original trained model:")
    evaluate(model, dataset.test)  # evaluate accuracy on test set

    model.transform(config="configs/corsair_mnist_lenet.yaml")  # transform model with corsair-specific features
    
    print(f"\nEvaluation of original model Corsair-transformed:")
    evaluate(model, dataset.test)  # evaluate accuracy on test set, again

    model = fine_tune(model, dataset.train)  # fine-tune model params

    print(f"\nEvaluation of Corsair-transformed model fine-tuned:")
    evaluate(model, dataset.test)  # evaluate accuracy on test set, again
