import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, norm=True, activate=True) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        )
        if norm:
            self.model.append(nn.BatchNorm2d(out_channel))
        if activate:
            self.model.append(nn.ReLU(inplace=True))

    def forward(self, x):
        return self.model(x)
    

class dncnn(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, hid_channel=64, kernel_size=3, padding=1, depth=20, **kwargs):
        super().__init__()
        layers = [
            block(in_channel, hid_channel, kernel_size, padding, False, True),
        ]
        for i in range(1, depth-1):
            layers.append(
                block(hid_channel, hid_channel, kernel_size, padding, True, True),
            )
        layers.append(
            block(hid_channel, out_channel, kernel_size, padding, False, False),
        )

        self.model = nn.Sequential(*layers)
        self.loss = nn.MSELoss()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


    def forward(self, x):
        return x + self.model(x)
    
    def training_loss(self, x, y, **kwargs):
        out = self(x)
        loss = self.loss(out, y)
        return loss

    def test(self, x, **kwargs):
        return self(x)