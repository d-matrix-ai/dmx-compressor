import torch, pytest, pickle
import numpy as np
from mltools import dmx, numerical
from numerics import TestData, TestOpData, convert_matmul_results


# helper dictionary that maps a numerics format string to a numerical.Format object
format_dict = {
    ("FP16", None): dmx.format.FLOAT16,
    ("FP32", None): dmx.format.FLOAT32,
    ("BFP32_1", None): dmx.format.FLOAT32,
    ("BFP16_64", "row"): numerical.Format.from_shorthand("BFP[8|8]{64,-1}(_N)"),
    ("BFP16_64", "col"): numerical.Format.from_shorthand("BFP[8|8]{64,-2}(_N)"),
    ("BFP12_128", "row"): numerical.Format.from_shorthand("BFP[4|8]{128,-1}(_N)"),
    ("BFP12_128", "col"): numerical.Format.from_shorthand("BFP[4|8]{128,-2}(_N)"),
    ("BFP24_64", "row"): numerical.Format.from_shorthand("BFP[16|8]{64,-1}(_N)"),
    ("BFP24_64", "col"): numerical.Format.from_shorthand("BFP[16|8]{64,-2}(_N)"),
    ("CFP[1|5|2]{15}(N)", None): numerical.Format.from_shorthand("FP[1|5|2,15](FN)"),
    ("CFP[1|5|2]{20}(N)", None): numerical.Format.from_shorthand("FP[1|5|2,20](FN)"),
    ("CFP[1|4|3]{7}(N)", None): numerical.Format.from_shorthand("FP[1|4|3,7](FN)"),
    ("CFP[1|4|3]{10}(N)", None): numerical.Format.from_shorthand("FP[1|4|3,10](FN)"),
    ("SBFP12", None): numerical.Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,7](FN)>{16,0}"
    ),
    ("UFP[0|4|4]{9}(N)", None): numerical.Format.from_shorthand("FP[0|4|4,9](FN)"),
}


@pytest.mark.parametrize(
    "from_format,to_format,register,rounding",
    (
        ("BFP32_1", "FP16", None, None),
        pytest.param(
            "FP16",
            "BFP16_64",
            "row",
            "nr-sym",
            marks=pytest.mark.xfail(reason="missing test data"),
        ),
        pytest.param(
            "FP16",
            "BFP16_64",
            "col",
            "nr-sym",
        ),
        pytest.param(
            "FP16",
            "BFP12_128",
            "row",
            "nr-sym",
            marks=pytest.mark.xfail(reason="missing test data"),
        ),
        pytest.param(
            "FP16",
            "BFP12_128",
            "col",
            "nr-sym",
        ),
        ("BFP16_64", "FP16", None, None),
        ("BFP12_128", "FP16", None, None),
        ("FP16", "BFP32_1", None, None),
        pytest.param(
            "FP16",
            "BFP24_64",
            "row",
            None,
            marks=pytest.mark.xfail(reason="qtorch bug, issue 75"),
        ),
        pytest.param(
            "FP16",
            "BFP24_64",
            "col",
            None,
            marks=pytest.mark.xfail(
                reason="Regression: this is a new failure, should be investigated, issue: 119"
            ),
        ),
        ("BFP24_64", "FP16", None, None),
        ("FP32", "CFP[1|5|2]{15}(N)", None, None),
        ("FP32", "CFP[1|5|2]{20}(N)", None, None),
        pytest.param(
            "FP32",
            "CFP[1|4|3]{7}(N)",
            None,
            None,
            marks=pytest.mark.skip(
                reason="likely due to different handling of inf, to be fixed"
            ),
        ),
        ("FP32", "CFP[1|4|3]{10}(N)", None, None),
        ("CFP[1|5|2]{15}(N)", "FP32", None, None),
        ("CFP[1|5|2]{20}(N)", "FP32", None, None),
        ("CFP[1|4|3]{7}(N)", "FP32", None, None),
        ("CFP[1|4|3]{10}(N)", "FP32", None, None),
        pytest.param(
            "SBFP12",
            "BFP16_64",
            None,
            None,
            marks=pytest.mark.skip(
                reason="likely due to undesired component uFP behavior, to be investigated"
            ),
        ),
        ("FP32", "UFP[0|4|4]{9}(N)", None, None),
        pytest.param(
            "FP16",
            "BFP16_64",
            "row",
            "u",
            marks=pytest.mark.xfail(
                reason="likely due to symmetric rounding of Mltools, to be investigated, issue 91"
            ),
        ),
        pytest.param(
            "FP16",
            "BFP16_64",
            "col",
            "u",
            marks=pytest.mark.xfail(
                reason="likely due to symmetric rounding of Mltools, to be investigated, issue 91"
            ),
        ),
        pytest.param(
            "FP16",
            "BFP12_128",
            "row",
            "u",
            marks=pytest.mark.xfail(
                reason="likely due to symmetric rounding of Mltools, to be investigated, issue 91"
            ),
        ),
        pytest.param(
            "FP16",
            "BFP12_128",
            "col",
            "u",
            marks=pytest.mark.xfail(
                reason="likely due to symmetric rounding of Mltools, to be investigated, issue 91"
            ),
        ),
        pytest.param(
            "FP16",
            "BFP16_64",
            "row",
            "d",
            marks=pytest.mark.xfail(
                reason="likely due to symmetric rounding of Mltools, to be investigated, issue 91"
            ),
        ),
        pytest.param(
            "FP16",
            "BFP16_64",
            "col",
            "d",
            marks=pytest.mark.xfail(
                reason="likely due to symmetric rounding of Mltools, to be investigated, issue 91"
            ),
        ),
        pytest.param(
            "FP16",
            "BFP12_128",
            "row",
            "d",
            marks=pytest.mark.xfail(
                reason="likely due to symmetric rounding of Mltools, to be investigated, issue 91"
            ),
        ),
        pytest.param(
            "FP16",
            "BFP12_128",
            "col",
            "d",
            marks=pytest.mark.xfail(
                reason="likely due to symmetric rounding of Mltools, to be investigated, issue 91"
            ),
        ),
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

    output_format = format_dict[(to_format, register)]
    fakeQ = dmx.CastTo(format=output_format)
    output = fakeQ(input_fp32)
    assert torch.allclose(output, output_fp32, atol=0)
    Q, dQ = numerical.Quantize.from_float(fakeQ), numerical.DeQuantize.from_float(fakeQ)
    _output = dQ(Q(input_fp32))
    assert torch.allclose(output, _output, atol=0)


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

    test_data = TestOpData("multiply", input_a_format, input_b_format)
    input_a_fp32 = torch.from_numpy(test_data.input1.to_fp32())
    input_b_fp32 = torch.from_numpy(test_data.input2.to_fp32())
    output_fp32 = torch.from_numpy(convert_matmul_results(test_data.output).to_fp32())

    output = torch.matmul(input_a_fp32, input_b_fp32)
    assert torch.allclose(output, output_fp32)
