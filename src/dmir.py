from utils.dmir_pb2 import *

# identifier strings of operators, compatible with ONNX naming
CASTTO              = "CastTo"
ADD                 = "Add"
MUL                 = "Mul"
TRANSPOSE           = "Transpose"
MATMUL              = "MatMul"
CONV2D              = "Conv"
AVGPOOL2D           = "AveragePool"
MAXPOOL2D           = "MaxPool"
BATCHNORM2D         = "BatchNormalization"
LAYERNORM           = "LayerNormalization"
DROPOUT             = "Dropout"
SOFTMAX             = "Softmax"
RELU                = "Relu"
RELU6               = "Relu6"
TANH                = "Tanh"
