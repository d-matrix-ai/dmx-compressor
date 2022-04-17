import torch
from mltools import corsair
from mltools.numerical import Format
from mlreferences import bert_base as ref


RANDOM_SEED = 0
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

corsair.aware()

bfp16 = Format.from_shorthand("BFP[8|8]{64,-1}(N)")
bfp12 = Format.from_shorthand("BFP[4|8]{128,-1}(N)")
same = Format.from_shorthand("SAME")

def test_transform():    
    expected_transforms = {}  # populate based on what we want to accomplish with the config file
    expected_transforms["model.body.bert.encoder.layer[0].attention.self.query"] = [bfp12, bfp16]
    expected_transforms["model.body.bert.encoder.layer[0].attention.self.value"] = [bfp12, bfp16]
    expected_transforms["model.body.bert.encoder.layer[5].attention.self.query"] = [bfp12, bfp16]
    expected_transforms["model.body.bert.encoder.layer[5].attention.self.key"] = [bfp12, bfp16]
    expected_transforms["model.body.bert.encoder.layer[0].intermediate.dense"] = [bfp16, bfp12]
    expected_transforms["model.body.bert.encoder.layer[5].intermediate.dense"] = [bfp16, bfp12]

    model = corsair.Model(ref.Net(ref.data_dir, BATCH_SIZE))
    model.transform("configs/corsair_bert_test.yaml")
    
    for layer in expected_transforms:
        assert(expected_transforms[layer][0].precision == eval(layer).input_cast.format.precision)
        assert(expected_transforms[layer][1].precision == eval(layer).weight_cast.format.precision)


