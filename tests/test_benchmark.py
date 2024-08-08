import triton
import torch
import sys

sys.path.insert(0, '../src')

from dmx.compressor.pt2bfp.quant.triton.bfp_ops import _quantize_bfp
from dmx.compressor import numerical

from dmx.compressor.numerical.format import Format
RANDOM_SEED = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'bfp_format_str'],  # Argument names to use as an x-axis for the plot
        x_vals=[[2048, 2048, "BFP[8|8]{64}(SU)"],[1024, 2048, "BFP[8|8]{32}(SD)"], [1024, 2048, "BFP[8|8]{128}(NU)"],[64, 64, "BFP[8|8]{64}(NU)"],
                [64, 64, "BFP[4|8]{64}(NU)"], [2048, 2048, "BFP[4|8]{64}(NU)"], [128, 128, "BFP[4|8]{64}(NU)"], [2048, 2048, "BFP[4|8]{256}(NU)"],
                [2048, 2048, "BFP[4|8]{1024}(NU)"], [2048, 2048, "BFP[4|8]{2048}(SU)"]],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cuda', 'triton'],
        # Label name for the lines
        line_names=["CUDA", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="quantization-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, bfp_format_str, provider):
    a = torch.randint(low = -128, high = 127, size = (M, N), device='cuda', dtype=torch.float).contiguous()
    quantiles = [0.5, 0.2, 0.8]
    elem_format_str = bfp_format_str
    elem_format = Format.from_shorthand(elem_format_str)
    rounding = elem_format.rounding
    symmetric = elem_format.symmetric
    block_size = elem_format.block_size
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: _quantize_bfp(a, 16, elem_format,
                      axes=[-1],
                      block_size=block_size,
                      round=rounding,
                      symmetric=symmetric,
                      format_str=elem_format_str,
                      custom_cuda=False), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: _quantize_bfp(a, 16, elem_format,
                      axes=[-1],
                      block_size=block_size,
                      round=rounding,
                      symmetric=symmetric,
                      format_str=elem_format_str,
                      custom_cuda=True), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * 1e-12 / (ms * 1e-3)
    # return ms, max_ms, min_ms
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=False, print_data=True, save_path='./result/quantization/')