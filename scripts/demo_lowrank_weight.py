import torch
from mltools import *
from mlreferences import lenet_512_512 as ref

dmir_dir = "/tools/d-matrix/ml/models/"
model_name = "lenet_512_512"

RANDOM_SEED = 0
BATCH_SIZE = 100

corsair.aware()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

dl = ref.data_loader(ref.data_dir, BATCH_SIZE)
x, _ = next(iter(dl))
sample_input = x.to(device)

model = corsair.Model(
    ref.Net((512, 512)),
    head=ref.Pre(),
).to(device)

sample_output0 = model(sample_input)

for rank in (512, 256, 128, 64, 32, 16, 8):
    model.body.intermediate_layers[0]._transform(
        dict(approximation_function=f"LOWRANK_WEIGHT(svd,{rank})")
    )
    sample_output1 = model(sample_input)
    print(f"Rank = {rank}: Frobenius norm of output error divided by Frobenius norm of original output = {torch.norm(sample_output1 - sample_output0) / torch.norm(sample_output0)}")
