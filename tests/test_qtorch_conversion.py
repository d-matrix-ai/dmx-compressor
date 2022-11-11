import torch, pytest, pickle
import numpy as np
from mltools import corsair


try:
    from numerics import TestData, UnaryNumericRnd

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize(
        "from_format,to_format,dimension",
        (
            ("BFP16_64", "FP16", None),
            ("BFP12_128", "FP16", None),
            ("BFP32_1", "BFP16_64", "row"),
            ("FP16", "BFP12_128", "row"),
            ("BFP24_64", "FP16", None),
            ("CFP[1|5|2]{15}(N)", "FP32", None),
            ("CFP[1|5|2]{20}(N)", "FP32", None),
            ("CFP[1|4|3]{7}(N)", "FP32", None),
            ("CFP[1|4|3]{10}(N)", "FP32", None),
            pytest.param(
                "FP16",
                "BFP24_64",
                None,
                marks=pytest.mark.xfail(reason="qtorch bug, issue 171"),
            ),
            pytest.param(
                "BFP32_1",
                "FP16",
                None,
                marks=pytest.mark.xfail(reason="qtorch bug, issue 171"),
            ),
            pytest.param(
                "FP16",
                "BFP16_64",
                "row",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 171"),
            ),
            pytest.param(
                "FP16",
                "BFP16_64",
                "col",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 171"),
            ),
            pytest.param(
                "FP16",
                "BFP12_128",
                "col",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 171"),
            ),
        ),
    )
    def test_conversion(from_format, to_format, dimension):
        """
        dimension specifies which dimension is used for blocking when to_format is BFP
        otherwise it should be set None
        """
        try:
            data = TestData(
                from_format, to_format, dimension if dimension == "col" else None
            )
        except (UnboundLocalError, ValueError) as e:
            assert False, f"lack of data: {from_format} to {to_format}"

        x = data.input.to_fp32()
        y = data.output.to_fp32()
        x, y = torch.Tensor(x).to(DEVICE), torch.Tensor(y).to(DEVICE)

        if to_format == "FP16":
            shorthand = "FP[1|5|10]{15}(FN)"

        elif to_format == "FP32":
            shorthand = "FP[1|8|23]{127}(FN)"

        elif to_format.startswith("BFP"):
            nbits = int(to_format.split("_")[0][3:])  # number of bits
            block_size, mantissa = int(to_format.split("_")[1]), nbits - 8
            dim_arg = 1 if dimension == "row" else "0"
            shorthand = (
                f"BFP[{mantissa}|8]" + "{" + f"{block_size},{dim_arg}" + "}" + "(N)"
            )
        qtorch_y = corsair.CastTo(shorthand)(x)
        mismatch = (
            qtorch_y - y
        ) != 0  # mismatch is a boolean arrary of the same size as that of x and y

        # detailed information of mismatched values
        err_msg = "values that qtorch disagrees with matlab conversion:\n"
        err_msg += f"Original Value in {from_format}:\n"
        err_msg += (
            f"{x[mismatch]}\n"  # extract all values from x which cause the mismatch
        )

        err_msg += f"Desired values (matlab converter) in {to_format}:\n"
        err_msg += f"{y[mismatch]}\n"

        err_msg += f"qtorch converted values in {to_format}:\n"
        err_msg += f"{qtorch_y[mismatch]}\n"

        assert torch.allclose(qtorch_y, y, atol=0), err_msg

    @pytest.mark.parametrize(
        "from_format,to_format,dimension,rnd",
        (
            pytest.param(
                "FP16",
                "BFP16_64",
                None,
                "u",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 207"),
            ),
            pytest.param(
                "FP16",
                "BFP16_64",
                "col",
                "u",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 207"),
            ),
            pytest.param(
                "FP16",
                "BFP12_128",
                None,
                "u",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 207"),
            ),
            pytest.param(
                "FP16",
                "BFP12_128",
                "col",
                "u",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 207"),
            ),
            pytest.param(
                "FP16",
                "BFP16_64",
                None,
                "d",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 207"),
            ),
            pytest.param(
                "FP16",
                "BFP16_64",
                "col",
                "d",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 207"),
            ),
            pytest.param(
                "FP16",
                "BFP12_128",
                None,
                "d",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 207"),
            ),
            pytest.param(
                "FP16",
                "BFP12_128",
                "col",
                "d",
                marks=pytest.mark.xfail(reason="qtorch bug, issue 207"),
            ),
        ),
    )
    def test_ud_conversion(from_format, to_format, dimension, rnd):
        """
        dimension specifies which dimension is used for blocking when to_format is BFP
        otherwise it should be set None
        """
        try:
            data = TestData(
                from_format, to_format, dimension if dimension == "col" else None
            )
        except (UnboundLocalError, ValueError) as e:
            assert False, f"lack of data: {from_format} to {to_format}"

        # the notions of U/D rounding might be different for positive and negative numbers. this can be changed if needed
        output_u = UnaryNumericRnd(from_format, to_format, dimension).convert(
            data.input, "u"
        )
        output_d = UnaryNumericRnd(from_format, to_format, dimension).convert(
            data.input, "d"
        )

        y_u, y_d = output_u.to_fp32(), output_d.to_fp32()

        x = data.input.to_fp32()
        if rnd == "u":
            y = np.where(x >= 0, y_u, y_d)
        else:
            y = np.where(x >= 0, y_d, y_u)

        x, y = torch.Tensor(x).to(DEVICE), torch.Tensor(y).to(DEVICE)

        if to_format == "FP16":
            shorthand = "FP[1|5|10]{15}(F" + ("U" if rnd == "u" else "D") + ")"

        elif to_format == "FP32":
            shorthand = "FP[1|8|23]{127}(F" + ("U" if rnd == "u" else "D") + ")"

        elif to_format.startswith("BFP"):
            nbits = int(to_format.split("_")[0][3:])  # number of bits
            block_size, mantissa = int(to_format.split("_")[1]), nbits - 8
            dim_arg = 1 if dimension == "row" else "0"
            shorthand = (
                f"BFP[{mantissa}|8]"
                + "{"
                + f"{block_size},{dim_arg}"
                + "}"
                + "("
                + ("U" if rnd == "u" else "D")
                + ")"
            )
        qtorch_y = corsair.CastTo(shorthand)(x)
        mismatch = (
            qtorch_y - y
        ) != 0  # mismatch is a boolean arrary of the same size as that of x and y

        # detailed information of mismatched values
        err_msg = "values that qtorch disagrees with matlab conversion:\n"
        err_msg += f"Original Value in {from_format}:\n"
        err_msg += (
            f"{x[mismatch]}\n"  # extract all values from x which cause the mismatch
        )

        err_msg += f"Desired values (matlab converter) in {to_format}:\n"
        err_msg += f"{y[mismatch]}\n"

        err_msg += f"qtorch converted values in {to_format}:\n"
        err_msg += f"{qtorch_y[mismatch]}\n"

        assert torch.allclose(qtorch_y, y, atol=0), err_msg

except ImportError:
    pass
