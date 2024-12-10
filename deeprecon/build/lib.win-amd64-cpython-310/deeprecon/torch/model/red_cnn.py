import torch.nn as nn


class red_cnn(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, hid_channel=48, kernel_size=5, padding=2, **kwargs):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, hid_channel, kernel_size=kernel_size, padding=padding)
        self.conv_2 = nn.Conv2d(hid_channel, hid_channel, kernel_size=kernel_size, padding=padding)
        self.conv_3 = nn.Conv2d(hid_channel, hid_channel, kernel_size=kernel_size, padding=padding)
        self.conv_4 = nn.Conv2d(hid_channel, hid_channel, kernel_size=kernel_size, padding=padding)
        self.conv_5 = nn.Conv2d(hid_channel, hid_channel, kernel_size=kernel_size, padding=padding)
        self.conv_t_1 = nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=kernel_size, padding=padding)
        self.conv_t_2 = nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=kernel_size, padding=padding)
        self.conv_t_3 = nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=kernel_size, padding=padding)
        self.conv_t_4 = nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=kernel_size, padding=padding)
        self.conv_t_5= nn.ConvTranspose2d(hid_channel, out_channel, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
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
        # encoder
        residual_1 = x.clone()
        out = self.relu(self.conv_1(x))
        out = self.relu(self.conv_2(out))
        residual_2 = out.clone()
        out = self.relu(self.conv_3(out))
        out = self.relu(self.conv_4(out))
        residual_3 = out.clone()
        out = self.relu(self.conv_5(out))

        # decoder
        out = self.conv_t_1(out)
        out = out + residual_3
        out = self.conv_t_2(self.relu(out))
        out = self.conv_t_3(self.relu(out))
        out = out + residual_2
        out = self.conv_t_4(self.relu(out))
        out = self.conv_t_5(self.relu(out))
        out = out + residual_1
        out = self.relu(out)
        return out

    def training_loss(self, x, y, **kwargs):
        out = self(x)
        loss = self.loss(out, y)
        return loss

    def test(self, x, **kwargs):
        return self(x)