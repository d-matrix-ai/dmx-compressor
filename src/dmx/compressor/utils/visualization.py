import math
import torch
import pptree
import dmx.compressor


def mask2braille(m, dims=(0, 1), max_elems=None):
    # permute dimensions
    ds = list(range(m.dim()))
    ds[0], ds[dims[0]] = dims[0], ds[0]
    ds[1], ds[dims[1]] = dims[1], ds[1]
    m = m.permute(*ds)
    truncate_h = truncate_w = False
    if max_elems is not None:
        truncate_h = max_elems < m.size(0)
        truncate_w = max_elems < m.size(1)
        m = m[:max_elems, :max_elems, ...]
    # reduce mask
    _h, _w = m.size(0), m.size(1)
    # m = m.reshape(_h, _w, -1).sum(-1) > 0
    m = m.reshape(_h, _w, -1)[:, :, 0]
    # pad mask
    h, w = math.ceil(_h / 4) * 4, math.ceil(_w / 2) * 2
    m = torch.nn.functional.pad(m, (0, w - _w, 0, h - _h))
    # compute values
    k = torch.Tensor([[0, 3], [1, 4], [2, 5], [6, 7]])
    x = (
        torch.nn.functional.conv2d(
            m.unsqueeze(0).unsqueeze(0).float(),
            2.0 ** k.to(m.device).unsqueeze(0).unsqueeze(0),
            stride=(4, 2),
        )
        .squeeze(0)
        .squeeze(0)
        .int()
        + 10240
    )
    x = "\n".join(["".join([chr(_x) for _x in row]) for row in x])
    return _box_wrap(
        x, pad_left=False, pad_right=truncate_w, pad_top=False, pad_bottom=truncate_h
    )


def _box_wrap(
    mls: str,
    pad_left: bool = False,
    pad_right: bool = False,
    pad_top: bool = False,
    pad_bottom: bool = False,
) -> str:
    lines = mls.split("\n")
    pad_char = "\u2592"
    if pad_left:
        lines = [pad_char + ln for ln in lines]
    if pad_right:
        lines = [ln + pad_char for ln in lines]
    if pad_top:
        lines = [pad_char * len(lines[0])] + lines
    if pad_bottom:
        lines = lines + [pad_char * len(lines[-1])]
    nchar = len(lines[0])
    lines = ["\u2502" + ln + "\u2502" for ln in lines]
    lines = (
        ["\u256d" + "\u2500" * nchar + "\u256e"]
        + lines
        + ["\u2570" + "\u2500" * nchar + "\u256f"]
    )
    return "\n".join(lines)


def print_model_tree(model: torch.nn.Module, include_type=False) -> str:
    node_str = lambda _n, _m: f"{_n}:{type(_m).__name__}" if include_type else _n
    is_leaf_node = compressor.dmx.nn.is_configurable
    m_root = pptree.Node(node_str("model", model))

    def get_tree(mod, root):
        for n, m in mod.named_children():
            n_ = pptree.Node(node_str(n, m), root)
            if not is_leaf_node(m):
                get_tree(m, n_)

    get_tree(model, m_root)
    pptree.print_tree(m_root)
