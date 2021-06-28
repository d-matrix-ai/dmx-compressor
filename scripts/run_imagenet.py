import os
import argparse
from dotenv import load_dotenv
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Imagenet models deployment on Corsair"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnet50",
        help="model name (default: resnet50)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/corsair_cnns.yaml",
        help="corsair config file",
    )
    parser.add_argument(
        "-z", "--batch-size", type=int, default=256, help="batch size (default: 256)"
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
    "alexnet",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet169",
    "densenet161",
    "densenet201",
    "inception_v3",
    "googlenet",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]
pb_wrap = lambda it: tqdm(it, leave=False, dynamic_ncols=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with pb_wrap(val_loader) as loader:
            for i, (images, target) in enumerate(loader):
                images, target = images.to(device), target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                loader.set_postfix(
                    loss="{:4.2f}".format(loss.item()))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def train(train_loader, model, criterion):
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader))
    for epoch in range(1, args.epochs+1):
        model.train()
        with pb_wrap(train_loader) as loader:
            loader.set_description(f"Epoch {epoch}")
            for i, (images, target) in enumerate(loader):
                images, target = images.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loader.set_postfix(dict(
                    lr=f"{scheduler.get_last_lr()[0] :.8f}",
                    loss="\33[91m{:6.4f}\033[0m".format(loss),
                ))
                scheduler.step()
        top1acc, top5acc = validate(ds.val, model, torch.nn.CrossEntropyLoss().to(device))
    return model

if __name__ == "__main__":

    import corsair
    from data import I1K
    import torchvision as tv

    ds = I1K(data_dir=os.path.join(DATA_PATH, 'imagenet'))

    assert (
        args.model in MODEL_LIST
    ), f"unrecognized model {args.model}, supported models: \n{MODEL_LIST}"

    model = eval("tv.models."+args.model)(pretrained=True).to(device)
    model.transform(config=args.config)
    print(f"model: {args.model}; config: {args.config}")
    top1acc, top5acc = validate(ds.val, model, torch.nn.CrossEntropyLoss().to(device))
    model = train(ds.train, model, torch.nn.CrossEntropyLoss().to(device))
    top1acc, top5acc = validate(ds.val, model, torch.nn.CrossEntropyLoss().to(device))
    