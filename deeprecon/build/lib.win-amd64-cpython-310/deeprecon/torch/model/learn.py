import torch
import torch.nn as nn
from ..utils.utils import create_projector2d


class fidelity_module(nn.Module):
    def __init__(self, projector, views):
        super().__init__()
        self.projector = projector
        self.views = views
        self.weight = nn.parameter.Parameter(torch.zeros([]))

    def forward(self, input_data, proj):
        temp = self.projector.projection(input_data, self.views) - proj
        intervening_res = self.projector.backprojection(temp, self.views)
        out = input_data - self.weight * intervening_res
        return out


class iter_block(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel, kernel_size, padding, projector, views):
        super().__init__()
        self.block1 = fidelity_module(projector, views)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channel, hid_channel, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channel, out_channel, kernel_size=kernel_size, padding=padding)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, proj):
        tmp1 = self.block1(input_data, proj)
        tmp2 = self.block2(input_data)
        output = tmp1 + tmp2
        output = self.relu(output)
        return output


class learn(nn.Module):
    def __init__(self, block_num=50, in_channel=1, out_channel=1, hid_channel=48, kernel_size=5, padding=2, **kwargs):
        super().__init__()
        projector = create_projector2d(**kwargs)
        if "views" in kwargs and kwargs["views"] is not None:
            views = kwargs["views"]
        else:
            views = torch.arange(0, kwargs["num_views"]) * torch.pi * 2 / kwargs["num_views"]
        views = nn.parameter.Parameter(torch.FloatTensor(views), requires_grad=False)
        self.model = nn.ModuleList([iter_block(in_channel, out_channel, hid_channel, kernel_size, padding, projector, views) for i in range(block_num)])
        self.loss = nn.MSELoss()
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_data, proj):
        x = input_data
        for index, module in enumerate(self.model):
            x = module(x, proj)
        return x

    def training_loss(self, x, y, p, **kwargs):
        out = self(x, p)
        loss = self.loss(out, y)
        return loss

    def test(self, x, p, **kwargs):
        return self(x, p)