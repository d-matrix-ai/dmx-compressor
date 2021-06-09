import os
import argparse
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx
from data import CIFAR10, CIFAR100


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR10/100 models deployment on Corsair"
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        default="cifar10",
        help="dataset, cifar10 or cifar100 (default: cifar10)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnet20",
        help="model name (default: resnet20)",
    )
    parser.add_argument(
        "-z", "--batch-size", type=int, default=64, help="batch size (default: 64)"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
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
MODEL_LIST = [
    "mobilenetv2_x0_5",
    "mobilenetv2_x0_75",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "shufflenetv2_x0_5",
    "shufflenetv2_x1_0",
    "shufflenetv2_x1_5",
    "shufflenetv2_x2_0",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "vit_b16",
    "vit_b32",
    "vit_h14",
    "vit_l16",
    "vit_l32",
]


def evaluate(model, ds):
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in ds:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy : {100. * correct / total :.2f}%")


if __name__ == "__main__":

    import corsair

    # load dataset
    dataset = eval(args.dataset.upper())(
        data_dir=os.path.join(DATA_PATH, "cifar"),
        cuda=use_cuda,
        num_workers=4,
        train_batch_size=args.batch_size,
        test_batch_size=1000,
        shuffle=True,
    )
    # load model
    assert args.model in MODEL_LIST, f"unrecognized model {args.model}"
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        args.dataset + "_" + args.model,
        pretrained=True,
    )
    model.transform(config_file="configs/corsair_cnns.yaml") 
    print(model)
    
    evaluate(model, dataset.test)
