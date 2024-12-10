import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel, kernel_size, padding) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, hid_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channel, out_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)

class down_sample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)
    
class up_sample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, output_padding) -> None:
        super().__init__()
        self.pool = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.pool(x)

class fbp_conv_net(nn.Module):
    def __init__(
        self,
        in_channel=1, 
        out_channel=1, 
        hid_channel=64, 
        kernel_size=3, 
        padding=1, 
        output_padding=1,
        channel_mult="1,2,4,8,16",
        **kwargs,
    ) -> None:
        super().__init__()

        mults = []
        for mult in channel_mult.split(","):
            mults.append(int(mult))
        
        ch = in_channel
        input_block_chans = []
        self.input_blocks = nn.ModuleList()
        for level, mult in enumerate(mults[:-1]):
            if level == 0:
                layers = [
                    nn.Conv2d(ch, int(mult * hid_channel), kernel_size, padding=padding),
                    nn.BatchNorm2d(int(mult * hid_channel)),
                    nn.ReLU(inplace=True),
                ]
                ch = int(mult * hid_channel)
            else:
                layers = [down_sample()]
            layers.append(
                block(ch, int(mult * hid_channel), int(mult * hid_channel), kernel_size, padding),
            )            
            self.input_blocks.append(nn.Sequential(*layers))
            ch = int(mult * hid_channel)
            input_block_chans.append(ch)

        self.middle_block = nn.Sequential(
            down_sample(),
            block(ch, int(mults[-1] * hid_channel), int(mults[-1] * hid_channel), kernel_size, padding),
            up_sample(int(mults[-1] * hid_channel), ch, kernel_size, padding, output_padding)
        )

        self.output_blocks = nn.ModuleList()
        mults = mults[-2::-1]
        for i in range(len(mults)):
            mult = mults[i]
            ich = input_block_chans.pop()
            layers = [
                block(ch + ich, int(mult * hid_channel), int(mult * hid_channel), kernel_size, padding),
            ]
            if i == (len(mults) - 1):
                layers.append(
                    nn.Conv2d(int(mult * hid_channel), out_channel, kernel_size=1)
                )
            else:
                layers.append(
                    up_sample(int(mult* hid_channel), int(mults[i+1] * hid_channel), kernel_size, padding, output_padding)
                )
                ch = int(mults[i+1] * hid_channel)
            self.output_blocks.append(nn.Sequential(*layers))
        
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
        hs = []
        h = x

        for module in self.input_blocks:
            h = module(h)
            hs.append(h)

        h = self.middle_block(h)

        for module in self.output_blocks:
            h = torch.cat((h, hs.pop()), dim=1)
            h = module(h)

        return x + h
    
    def training_loss(self, x, y, **kwargs):
        out = self(x)
        loss = self.loss(out, y)
        return loss

    def test(self, x, **kwargs):
        return self(x)
        

