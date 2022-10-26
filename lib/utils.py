import functools
import numpy as np
import torch

def print_model(model):
    print('Parameters:')
    total_params = 0
    for name, param in model.named_parameters():
        print(f"\t{name}: {list(param.shape)}")
        if len(list(param.shape)) == 0:
            total_params += 1
        else:
            total_params += functools.reduce(
                (lambda x,y: x*y), list(param.shape))
    print(f'Total parameters: {total_params:,}')

def print_row(*row, colwidth=16):
    """Print a row of values."""
    def format_val(x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str( x).ljust(colwidth)[:colwidth]
    print("  ".join([format_val(x) for x in row]))