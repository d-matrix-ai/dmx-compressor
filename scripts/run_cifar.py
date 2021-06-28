import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx
from data import CIFAR10, CIFAR100
from utils import load_config_file, mask2braille


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
        "-c",
        "--config",
        type=str,
        default="configs/corsair_cnns.yaml",
        help="corsair config file",
    )
    parser.add_argument(
        "-z", "--batch-size", type=int, default=128, help="batch size (default: 128)"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 50)",
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
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "mobilenetv2_x0_5",
    "mobilenetv2_x0_75",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "shufflenetv2_x0_5",
    "shufflenetv2_x1_0",
    "shufflenetv2_x1_5",
    "shufflenetv2_x2_0",
]
pb_wrap = lambda it: tqdm(it, leave=False, dynamic_ncols=True)

def evaluate(model, ds):
    print(f"Evaluating {args.model} on {args.dataset}...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with pb_wrap(ds) as loader:
            for (images, labels) in loader:
                loader.set_description("Evaluation")
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    print(f"\tAccuracy : {100. * correct / total :.2f}%")

def train(model, ds, epochs):
    print(f"Training {args.model} on {args.dataset}...")
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(ds))
    for epoch in range(1, epochs+1):
        model.train()
        with pb_wrap(ds) as loader:
            loader.set_description(f"Epoch {epoch}")
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                loader.set_postfix(dict(
                    lr=f"{scheduler.get_last_lr()[0] :.8f}",
                    loss="\33[91m{:6.4f}\033[0m".format(loss),
                ))
                scheduler.step()
        evaluate(model, dataset.test)
    return model

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
    assert (
        args.model in MODEL_LIST
    ), f"unrecognized model {args.model}, supported models: \n{MODEL_LIST}"
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        args.dataset + "_" + args.model,
        pretrained=True,
    ).to(device)

    config = load_config_file(args.config)
    model.transform(config)
    # evaluate model
    evaluate(model, dataset.test)
    
    model = train(model, dataset.train, args.epochs)
    # for nnz in [7, 6, 5, 4]:
    #     config["transformation_rules"][0]["config"]["weight_sparseness"] = f"BTOPK{{{nnz}:8,1}}"
    #     model.transform(config)
    #     model = train(model, dataset.train, 10)
    # for nnz in [3, 2]:
    #     config["transformation_rules"][0]["config"]["weight_sparseness"] = f"BTOPK{{{nnz}:8,1}}"
    #     model.transform(config)
    #     model = train(model, dataset.train, 25)
    # evaluate model
    evaluate(model, dataset.test)
