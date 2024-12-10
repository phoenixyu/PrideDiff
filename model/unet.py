import torch
import math
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from inspect import isfunction

# def exists(x):
#     return x is not None

# def default(val, d):
#     if exists(val):
#         return val
#     return d() if isfunction(d) else d

# class LayerNorm(nn.Module):
#     def __init__(self, dim, eps = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
#         self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

#     def forward(self, x):
#         var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
#         mean = torch.mean(x, dim = 1, keepdim = True)
#         return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = LayerNorm(dim)

#     def forward(self, x):
#         x = self.norm(x)
#         return self.fn(x)

# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb
    
# class ConvNextBlock(nn.Module):
#     """ https://arxiv.org/abs/2201.03545 """

#     def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.GELU(),
#             nn.Linear(time_emb_dim, dim)
#         ) if exists(time_emb_dim) else None

#         self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

#         self.net = nn.Sequential(
#             LayerNorm(dim) if norm else nn.Identity(),
#             nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
#             nn.GELU(),
#             nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
#         )

#         self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb = None):
#         h = self.ds_conv(x)

#         if exists(self.mlp):
#             assert exists(time_emb), 'time emb must be passed in'
#             condition = self.mlp(time_emb)
#             h = h + rearrange(condition, 'b c -> b c 1 1')

#         h = self.net(h)
#         return h + self.res_conv(x)

# def Upsample(dim):
#     return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

# def Downsample(dim):
#     return nn.Conv2d(dim, dim, 4, 2, 1)

# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x

# class LinearAttention(nn.Module):
#     def __init__(self, dim, heads = 4, dim_head = 32):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim = 1)
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
#         q = q * self.scale

#         k = k.softmax(dim = -1)
#         context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

#         out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
#         out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
#         return self.to_out(out)

# class Unet(nn.Module):
#     def __init__(
#         self,
#         dim,
#         out_dim = None,
#         dim_mults=(1, 2, 4, 8),
#         channels = 1,
#         with_time_emb = True,
#         residual = False
#     ):
#         super().__init__()
#         self.channels = channels
#         self.residual = residual
#         print("Is Time embed used ? ", with_time_emb)

#         dims = [channels, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))

#         if with_time_emb:
#             time_dim = dim
#             self.time_mlp = nn.Sequential(
#                 SinusoidalPosEmb(dim),
#                 nn.Linear(dim, dim * 4),
#                 nn.GELU(),
#                 nn.Linear(dim * 4, dim)
#             )
#         else:
#             time_dim = None
#             self.time_mlp = None

#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)

#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)

#             self.downs.append(nn.ModuleList([
#                 ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
#                 ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
#                 Residual(PreNorm(dim_out, LinearAttention(dim_out))),
#                 Downsample(dim_out) if not is_last else nn.Identity()
#             ]))

#         mid_dim = dims[-1]
#         self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
#         self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
#         self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#             is_last = ind >= (num_resolutions - 1)

#             self.ups.append(nn.ModuleList([
#                 ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
#                 ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
#                 Residual(PreNorm(dim_in, LinearAttention(dim_in))),
#                 Upsample(dim_in) if not is_last else nn.Identity()
#             ]))

#         out_dim = default(out_dim, channels)
#         self.final_conv = nn.Sequential(
#             ConvNextBlock(dim, dim),
#             nn.Conv2d(dim, out_dim, 1)
#         )

#     def forward(self, x, time):
#         orig_x = x
#         t = self.time_mlp(time) if exists(self.time_mlp) else None

#         h = []

#         for convnext, convnext2, attn, downsample in self.downs:
#             x = convnext(x, t)
#             x = convnext2(x, t)
#             x = attn(x)
#             h.append(x)
#             x = downsample(x)

#         x = self.mid_block1(x, t)
#         x = self.mid_attn(x)
#         x = self.mid_block2(x, t)

#         for convnext, convnext2, attn, upsample in self.ups:
#             x = torch.cat((x, h.pop()), dim=1)
#             x = convnext(x, t)
#             x = convnext2(x, t)
#             x = attn(x)
#             x = upsample(x)
#         if self.residual:
#             return self.final_conv(x) + orig_x

#         return self.final_conv(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class adjust_net(nn.Module):
    def __init__(self, out_channels=64, middle_channels=32):
        super(adjust_net, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(2, middle_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(middle_channels, middle_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(middle_channels*2, middle_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(middle_channels*4, out_channels*2, 1, padding=0)
        )

    def forward(self, x):
        out = self.model(x)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out1 = out[:, :out.shape[1]//2]
        out2 = out[:, out.shape[1]//2:]
        return out1, out2

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(UNet, self).__init__()

        dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.inc = nn.Sequential(
            single_conv(in_channels, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.mlp1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.adjust1 = adjust_net(64)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.mlp2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.adjust2 = adjust_net(128)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.mlp3 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.adjust3 = adjust_net(128)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.mlp4 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.adjust4 = adjust_net(64)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, out_channels)

    def forward(self, x, t, x_adjust=None, adjust=False):
        inx = self.inc(x)
        time_emb = self.time_mlp(t)
        down1 = self.down1(inx)
        condition1 = self.mlp1(time_emb)
        b, c = condition1.shape
        condition1 = rearrange(condition1, 'b c -> b c 1 1')
        if adjust:
            gamma1, beta1 = self.adjust1(x_adjust)
            down1 = down1 + gamma1 * condition1 + beta1
        else:
            down1 = down1 + condition1
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        condition2 = self.mlp2(time_emb)
        b, c = condition2.shape
        condition2 = rearrange(condition2, 'b c -> b c 1 1')
        if adjust:
            gamma2, beta2 = self.adjust2(x_adjust)
            down2 = down2 + gamma2 * condition2 + beta2
        else:
            down2 = down2 + condition2
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        condition3 = self.mlp3(time_emb)
        b, c = condition3.shape
        condition3 = rearrange(condition3, 'b c -> b c 1 1')
        if adjust:
            gamma3, beta3 = self.adjust3(x_adjust)
            up1 = up1 + gamma3 * condition3 + beta3
        else:
            up1 = up1 + condition3
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        condition4 = self.mlp4(time_emb)
        b, c = condition4.shape
        condition4 = rearrange(condition4, 'b c -> b c 1 1')
        if adjust:
            gamma4, beta4 = self.adjust4(x_adjust)
            up2 = up2 + gamma4 * condition4 + beta4
        else:
            up2 = up2 + condition4
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out
