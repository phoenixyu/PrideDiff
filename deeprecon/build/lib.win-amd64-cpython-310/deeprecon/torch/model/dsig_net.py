import torch
import numpy as np
import torch.nn as nn
from .iradon_map import back_proj_net
from ..utils.utils import create_projector2d


class down_sampling_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, scale_factor, relu_slope) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=scale_factor),
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding),
            nn.GroupNorm(out_channel, num_groups=1, affine=False),
            nn.LeakyReLU(relu_slope, True),
        )

    def forward(self, x):
        return self.model(x)
    

class up_sampling_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, scale_factor, relu_slope) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.PixelShuffle(upscale_factor=scale_factor),
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding),
            nn.GroupNorm(out_channel, num_groups=1, affine=False),
            nn.LeakyReLU(relu_slope, True),
        )

    def forward(self, x):
        return self.model(x)

class res_block(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel, kernel_size, padding, relu_slope) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, hid_channel, kernel_size, padding=padding),
            nn.GroupNorm(hid_channel, num_groups=1, affine=False),
            nn.LeakyReLU(relu_slope, True),
            nn.Conv2d(hid_channel, out_channel, kernel_size, padding=padding),
            nn.GroupNorm(out_channel, num_groups=1, affine=False),
        )
        self.relu = nn.LeakyReLU(relu_slope, True)

    def forward(self, x):
        return self.relu(x + self.model(x))
    

class sino_net(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel, kernel_size, padding, scale_factor, relu_slope, depth) -> None:
        super().__init__()
        num_filters = hid_channel * scale_factor ** 2
        layers = [
            nn.Conv2d(in_channel, hid_channel, kernel_size, padding=padding),
            nn.GroupNorm(hid_channel, num_groups=1, affine=False),
            nn.LeakyReLU(relu_slope, True),
            down_sampling_block(num_filters, num_filters, kernel_size, padding, scale_factor, relu_slope),
        ]
        for i in range(depth - 2):
            layers.append(res_block(num_filters, num_filters, num_filters, kernel_size, padding, relu_slope))
        layers.append(
            nn.Conv2d(num_filters, out_channel, 1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class spatial_net(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel, kernel_size, padding, scale_factor, relu_slope, depth) -> None:
        super().__init__()
        num_filters = hid_channel * scale_factor ** 2
        layers = [
            nn.Conv2d(in_channel, num_filters, kernel_size, padding=padding),
            nn.GroupNorm(num_filters, num_groups=1, affine=False),
            nn.LeakyReLU(relu_slope, True),            
        ]
        for i in range(depth - 2):
            layers.append(res_block(num_filters, num_filters, num_filters, kernel_size, padding, relu_slope))
        layers.append(
            up_sampling_block(hid_channel, hid_channel, kernel_size, padding, scale_factor, relu_slope)
        )
        layers.append(
            nn.Conv2d(hid_channel, out_channel, kernel_size=1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class dsig_net(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, hid_channel=16, bp_channel=4, kernel_size=3, padding=1, scale_factor=2, relu_slop=0.2, depth=14, learn=True, **kwargs) -> None:
        super().__init__()
        kwargs["image_size"] = kwargs["image_size"] // scale_factor if np.isscalar(kwargs["image_size"]) else [x // scale_factor for x in kwargs["image_size"]]
        kwargs["num_view"] = kwargs["num_view"] // scale_factor
        kwargs["num_det"] = kwargs["num_det"] // scale_factor
        kwargs["iso_source"] = kwargs["iso_source"] / scale_factor
        kwargs["source_detector"] = kwargs["source_detector"] / scale_factor
        kwargs["pixshift"] = kwargs["pixshift"] / scale_factor if np.isscalar(kwargs["pixshift"]) else [x / scale_factor for x in kwargs["pixshift"]]
        kwargs["binshift"] = kwargs["binshift"] / scale_factor
        projector = create_projector2d(**kwargs)
        if "views" in kwargs and kwargs["views"] is not None:
            views = kwargs["views"]
        else:
            views = torch.arange(0, kwargs["num_views"]) * torch.pi * 2 / kwargs["num_views"]

        image_size = projector.image_size
        sys_matrix = projector.get_sys_matrix(views)
        coef = (views[-1] - views[0]) / (2 * (views.shape[0] - 1))

        self.model = nn.Sequential(
            sino_net(in_channel, bp_channel, hid_channel, kernel_size, padding, scale_factor, relu_slop, depth),
            back_proj_net(sys_matrix, image_size, coef, learn),
            spatial_net(bp_channel, out_channel, hid_channel, kernel_size, padding, scale_factor, relu_slop, depth),
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
