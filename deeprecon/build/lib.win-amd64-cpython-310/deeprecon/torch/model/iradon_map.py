import torch
import numpy as np
import torch.nn as nn
from ..utils.utils import create_projector2d

class back_proj_net(nn.Module):
    def __init__(self, sys_matrix, image_size, coef, learn) -> None:
        super().__init__()
        sys_matrix = sys_matrix.t()
        if not sys_matrix.is_coalesced():
            sys_matrix = sys_matrix.coalesce()
        self.weight = nn.Parameter(sys_matrix, requires_grad=learn)
        if learn == True:
            self.bias = nn.parameter(torch.zeros(np.prod(image_size)))
        else:
            self.weight.values().fill_(1.)
            self.bias = None
        self.image_size = image_size
        self.coef = coef

    def forward(self, p):
        b, c, h, w = p.shape
        p = p.reshape(b * c, h * w)
        p = p.t()
        output = torch.sparse.mm(self.weight, p)
        output = output.t() * self.coef
        if self.bias is not None:
            output = output + self.bias
        output = output.reshape(b, c, *self.image_size)

        return output
    

class block(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel, kernel_size, padding) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, hid_channel, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channel, out_channel, kernel_size, padding=padding)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):        
        return self.relu(x + self.model(x))

class rcnn(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel, kernel_size, padding, depth) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channel, hid_channel, kernel_size, padding=padding),
            nn.GroupNorm(num_channels=hid_channel, num_groups=1, affine=False),
            nn.ReLU(),
        ]
        for i in range(depth - 2):
            layers.append(block(hid_channel, hid_channel, hid_channel, kernel_size, padding))
        layers.append(nn.Conv2d(hid_channel, out_channel, kernel_size, padding=padding))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class iradon_map(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, hid_channel=64, kernel_size=3, padding=1, depth=13, learn=True, **kwargs) -> None:
        super().__init__()        
        projector = create_projector2d(**kwargs)
        if "views" in kwargs and kwargs["views"] is not None:
            views = kwargs["views"]
        else:
            views = torch.arange(0, kwargs["num_views"]) * torch.pi * 2 / kwargs["num_views"]

        image_size = projector.image_size
        num_det = projector.num_det
        sys_matrix = projector.get_sys_matrix(views)
        coef = (views[-1] - views[0]) / (2 * (views.shape[0] - 1))
        self.model = nn.Sequential(
            nn.Linear(num_det, num_det),
            nn.ReLU(),
            nn.Linear(num_det, num_det),
            nn.ReLU(),
            back_proj_net(sys_matrix, image_size, coef, learn),
            rcnn(in_channel, out_channel, hid_channel, kernel_size, padding, depth),
        )
        self.loss = nn.MSELoss()

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, p):
        return self.model(p)

    def training_loss(self, p, y, **kwargs):
        out = self(p)
        loss = self.loss(out, y)
        return loss

    def test(self, p, **kwargs):
        return self(p)