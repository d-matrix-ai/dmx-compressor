import torch, pytest, pickle
import numpy as np
from mltools import corsair, numerical
from numerics import TestData, TestOpData


# helper dictionary that maps a numerics format string to a numerical.Format object
format_dict = {
    "FP16": corsair.format.FLOAT16,
    "FP32": corsair.format.FLOAT32,
    "BFP32_1": corsair.format.FLOAT32,
    "BFP16_64": corsair.format.BFP16_64_LD,
    "BFP12_128": corsair.format.BFP12_128_LD,
    "BFP24_64": numerical.Format.from_shorthand("BFP[16|8]{64,-1}(_N)"),
    "CFP[1|5|2]{15}(N)": numerical.Format.from_shorthand("FP[1|5|2,15](FN)"),
    "CFP[1|5|2]{20}(N)": numerical.Format.from_shorthand("FP[1|5|2,20](FN)"),
    "CFP[1|4|3]{7}(N)": numerical.Format.from_shorthand("FP[1|4|3,7](FN)"),
    "CFP[1|4|3]{10}(N)": numerical.Format.from_shorthand("FP[1|4|3,10](FN)"),
    "SBFP12": numerical.Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,7](FN)>{16,-1}"),
}


@pytest.mark.parametrize(
    "from_format,to_format,register,rounding",
    (
        ("BFP32_1", "FP16", None, None),
        ("FP16", "BFP16_64", "row", None),
        ("FP16", "BFP16_64", "col", None),
        ("FP16", "BFP12_128", "row", None),
        ("FP16", "BFP12_128", "col", None),
        ("BFP16_64", "FP16", None, None),
        ("BFP12_128", "FP16", None, None),
        ("FP16", "BFP32_1", None, None),
        ("FP16", "BFP24_64", "row", None),
        ("FP16", "BFP24_64", "col", None),
        ("BFP24_64", "FP16", None, None),
        ("FP32", "CFP[1|5|2]{15}(N)", None, None),
        ("FP32", "CFP[1|5|2]{20}(N)", None, None),
        ("FP32", "CFP[1|4|3]{7}(N)", None, None),
        ("FP32", "CFP[1|4|3]{10}(N)", None, None),
        ("CFP[1|5|2]{15}(N)", "FP32", None, None),
        ("CFP[1|5|2]{20}(N)", "FP32", None, None),
        ("CFP[1|4|3]{7}(N)", "FP32", None, None),
        ("CFP[1|4|3]{10}(N)", "FP32", None, None),
        ("SBFP12", "BFP16_64", None, None),
    ),
)
def test_conversion(from_format, to_format, register, rounding):
    """
    Conversion of one tensor format to another
    the test assures mltools conversion is accurate
    """

    test_data = TestData(from_format, to_format, register, rounding)
    input_fp32 = torch.from_numpy(test_data.input.to_fp32())
    output_fp32 = torch.from_numpy(test_data.output.to_fp32())

    output = corsair.CastTo(format=format_dict[to_format])(input_fp32)
    assert torch.allclose(output, output_fp32, atol=0)


@pytest.mark.parametrize(
    "input_a_format,input_b_format",
    (
        ("BFP16_64", "BFP16_64"),
        ("BFP16_64", "BFP12_128"),
        ("BFP12_128", "BFP12_128"),
    ),
)
def test_matmul(input_a_format, input_b_format):
    """
    Matmul between two tensors of custom format
    the test assures torch.matmul() is epsilon-accurate
    """

    test_data = TestOpData(input_a_format, input_b_format)
    input_a_fp32 = torch.from_numpy(test_data.input_a.to_fp32())
    input_b_fp32 = torch.from_numpy(test_data.input_b.to_fp32())
    output_fp32 = torch.from_numpy(test_data.output.to_fp32())

    output = torch.matmul(input_a_fp32, input_b_fp32)
    assert torch.allclose(output, output_fp32)
