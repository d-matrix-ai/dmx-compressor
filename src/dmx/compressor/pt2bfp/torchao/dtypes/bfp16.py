import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch.library import impl, Library


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


def fill_defaults(args, n, defaults_tail):
    """
    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.
    Args:
        args: the list of positional arguments passed to __torch_dispatch__
        n: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list
    Example:
        >>> fill_defaults([1, 2, 3], 5, [3, 4, 5])
        [1, 2, 3, 4, 5]
        >>> fill_defaults([1, 2, 3], 5, [None, None, None])
        [1, 2, 3, None, None]]
    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError("not enough defaults to fill arguments")
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i - n + len(defaults_tail)])
    return r


# from
# https://github.com/drisspg/transformer_nuggets/blob/9ad3a7fc552a954eb702ade0e276b8d8e09c3db6/transformer_nuggets/quant/qlora.py#L233


def unpack_bfp16(float_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using uint8 we will decode 2 entries per byte
    # Shift elements down 4 and select out the bottom 4 bits
    return float_data


def pack_bfp16(float_data) -> torch.Tensor:
    # converting to uint8 for operations
    return float_data


qtensor_lib = Library("qtensors", "DEF")
qtensor_lib.define(
    "quantize_per_tensor_bfp16(Tensor input, float scale, int zero_point) -> Tensor"
)


@impl(qtensor_lib, "quantize_per_tensor_bfp16", "CompositeExplicitAutograd")
def quantize_per_tensor_bfp16(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    inv_scale = 1.0 / scale
    return pack_bfp16(
        input
    )


qtensor_lib.define(
    "dequantize_per_tensor_bfp16(Tensor input, float scale, int zero_point) -> Tensor"
)


@impl(qtensor_lib, "dequantize_per_tensor_bfp16", "CompositeExplicitAutograd")
def dequantize_per_tensor_bfp16(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    input = unpack_bfp16(input)
    return input


class BFP16Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, **kwargs):
        assert elem.dtype is torch.float
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False

        return torch.Tensor._make_wrapper_subclass(
            cls, up_size(elem.shape), dtype=torch.float, **kwargs
        )

    def __init__(self, elem, **kwargs):

        self.elem = elem

    @classmethod
    def from_unpacked(cls, unpacked):
        return BFP16Tensor(pack_bfp16(unpacked))

    def tolist(self):
        return self.to(torch.float).tolist()

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(flattened, meta, outer_size, outer_stride):
        assert meta is None
        elem = flattened["elem"]
        return BFP16Tensor(elem)

    def __hash__(self):
        return hash(self.elem)

    def __eq__(self, other):
        return torch.equal(self.elem, other.elem)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is torch.ops.aten.view.default:
            self, size = args
            size = utils.infer_size(size, self.numel())
            assert not kwargs
            # WARNING: views not preserved
            return BFP16Tensor(self.elem.reshape(down_size(size)))
        elif func is torch.ops.aten.view.dtype:
            self, dtype = args
            if dtype == torch.float:
                return unpack_bfp16(self.elem).view(torch.float)
            return NotImplementedError(f"view {args}")
        elif func is torch.ops.aten.to.dtype:
            self, dtype = args
            if dtype == torch.float:
                return unpack_bfp16(self.elem).view(torch.float)
            return NotImplementedError(f"to {args}")
        elif func is torch.ops.aten.eq.Tensor:
            args = pytree.tree_map_only(
                BFP16Tensor, lambda x: x.elem.view(torch.float), args
            )
            kwargs = pytree.tree_map_only(
                BFP16Tensor, lambda x: x.elem.view(torch.float), kwargs
            )
            return torch.ops.aten.eq.Tensor(*args, **kwargs)
        elif func is torch.ops.aten._to_copy.default:
            (self,) = args
            if kwargs == {"dtype": torch.float}:
                return unpack_bfp16(self.elem).view(self.shape)  # no wrap
            else:
                raise NotImplementedError(f"_to_copy {kwargs}")
        elif func is torch.ops.aten.unbind.int:
            # This is tricky.  Given torch.tensor([0, 1, 2, 3]) we want to
            # create four tensors containing one element each.  But we can't
            # do this with uint4 because such a tensor's size is not divisible
            # by bytes.  What I am going to do instead is promote to uint8
            # when this happens
            self, dim = fill_defaults(args, 2, [0])
            if dim != self.dim() - 1:
                raise NotImplementedError(f"unbind dim={dim}")
            else:
                # We're unbinding the last dimension, need to promote
                return torch.ops.aten._to_copy.default(self, dtype=torch.float).unbind(
                    dim
                )
        elif func is torch.ops.aten.select.int:
            self, dim, index = args
            if dim != self.dim() - 1:
                return BFP16Tensor(torch.ops.aten.select.int(self.elem, dim, index))
            else:
                raise NotImplementedError(f"select dim={dim}")
        elif func is torch.ops.aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == self.dim() - 1:
                # hard case
                if step != 1:
                    raise NotImplementedError(f"slice step={step}")
                assert start % 2 == 0, start
                assert end >= self.shape[dim] or end % 2 == 0, end
                return BFP16Tensor(
                    torch.ops.aten.slice.Tensor(self.elem, dim, start // 2, end // 2, 1)
                )
            else:
                # easy case
                return BFP16Tensor(
                    torch.ops.aten.slice.Tensor(self.elem, dim, start, end, step)
                )
        elif func is torch.ops.aten.t.default:
            # assert False, "transpose is not properly implemented currently"
            (self,) = args
            unpacked = unpack_bfp16(self.elem)
            transposed = torch.ops.aten.t.default(unpacked)
            transposed_and_packed = pack_bfp16(transposed)
            return BFP16Tensor(transposed_and_packed)
        elif func is torch.ops.aten.transpose_copy.int:
            self, dim0, dim1 = args
            unpacked = unpack_bfp16(self.elem).view(self.shape)
            transposed = torch.ops.aten.transpose_copy.int(unpacked, dim0, dim1)
            transposed_and_packed = pack_bfp16(transposed)
            return BFP16Tensor(transposed_and_packed)
        elif func is torch.ops.aten.as_strided.default:
            # size, stride, storage_offset are referring to tensor elements, not physical bytes
            self, size, stride, storage_offset = args
            size = down_size(size)

            new_stride = []
            for s in stride:
                if s != 1:
                    # since two int4 equals to 1 uint8
                    new_stride.append(s // 2)
                else:
                    new_stride.append(s)
            stride = new_stride

            storage_offset //= 2
            return BFP16Tensor(
                torch.ops.aten.as_strided.default(
                    self.elem, size, stride, storage_offset
                )
            )
        elif func is torch.ops.aten.isnan.default:
            return False
        elif func is torch.ops.aten.abs.default:
            (self,) = args
            return BFP16Tensor(torch.ops.aten.abs.default(
                    self.elem
                ))
        elif func is torch.ops.aten.ne.Scalar:
            self, other = args
            return BFP16Tensor(torch.ops.aten.ne.Scalar(
                    self.elem, other
                ))
        elif func is torch.ops.aten.isclose.default:
            (self, self2, tol) = args
            out = self.isclose(self2, rtol=tol, atol=1e-08, equal_nan=False)
            return BFP16Tensor(torch.ops.aten.isclose.default(
                    self.elem, self2.elem, tol
                ))
        
                

        raise NotImplementedError(f"{func}")

    __torch_function__ = torch._C._disabled_torch_function_impl


