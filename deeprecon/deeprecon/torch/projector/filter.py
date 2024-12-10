import torch

__filters__ = {
    "ramp",
    "sheeplogan",
}

def filter_generate(num_det, det_size, filter_type="ramp", dtype=None):
    assert filter_type in __filters__, f"Filter type {filter_type} not supported"
    return eval(filter_type)(num_det, det_size, dtype)

def ramp(num_det, det_size, dtype):
    filter = torch.empty(num_det * 2 - 1, dtype=dtype)
    idx = torch.arange(num_det * 2 - 1)
    x = idx - num_det + 1
    if (abs(x[0]) % 2) == 1:
        filter[::2] = -1 / (torch.pi * torch.pi * det_size * det_size * (x[::2] ** 2))
        filter[1::2] = 0.        
    else:
        filter[::2] = 0.
        filter[1::2] = -1 / (torch.pi * torch.pi * det_size * det_size * (x[1::2] ** 2))
    filter[num_det - 1] = 1 / (4 * det_size * det_size)
    return filter

def sheeplogan(num_det, det_size, dtype):
    idx = torch.arange(num_det * 2 - 1)
    x = idx - num_det + 1
    filter = - 2 / (torch.pi * torch.pi * det_size * det_size * (4 * x * x - 1))
    filter = filter.to(dtype)
    return filter
