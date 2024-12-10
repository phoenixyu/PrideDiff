import torch
import torch.nn.functional as F
import math
from torch.autograd import Function

from torch import nn
from deeprecon.torch.projector import projector2d

dr_projector = projector2d(image_size=512, 
                        num_det=768, 
                        pix_size=0.7433, 
                        det_size=1.2858, 
                        iso_source=595.0, 
                        source_detector=490.6, scan_type="arc").cuda()

views = torch.arange(720) * torch.pi * 2 / 720
views = views.cuda()

A = dr_projector.projection
AT = dr_projector.filtered_backprojection

class prj_fun(Function):
    @staticmethod
    def forward(self, image):
        return dr_projector.projection(image, views)

    @staticmethod
    def backward(self, grad_output):
        grad_input = dr_projector.filtered_backprojection(grad_output.contiguous(), views)
        return grad_input

class prj_t_fun(Function):
    @staticmethod
    def forward(self, proj):
        return dr_projector.filtered_backprojection(proj, views)

    @staticmethod
    def backward(self, grad_output):
        grad_input = dr_projector.projection(grad_output.contiguous(), views)
        return grad_input

class projector(nn.Module):
    def __init__(self):
        super(projector, self).__init__()
        
    def forward(self, image):
        return prj_fun.apply(image)

class projector_t(nn.Module):
    def __init__(self):
        super(projector_t, self).__init__()
        
    def forward(self, proj):
        return prj_t_fun.apply(proj)

class fidelity_module(nn.Module):
    def __init__(self):
        super(fidelity_module, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1).squeeze())
        self.projector = projector()
        self.projector_t = projector_t()
        
    def forward(self, input_data, proj, t):
        temp = self.projector(input_data) - proj
        intervening_res = self.projector_t(temp)
        # out = input_data - self.weight[t].view(input_data.shape[0], 1, 1, 1) * intervening_res
        out = input_data - self.weight * intervening_res
        return out

#### Prior Extraction Module #####
class REDCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ch = 1
        self.out_ch = 64

        self.l1 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, 3, padding=1, stride=1),
            nn.ReLU()
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1, stride=1),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1, stride=1),
            nn.ReLU()
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1, stride=1),
            nn.ReLU()
        )

        self.l5 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1, stride=1),
        )

        self.l6 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1, stride=1),
        )

        self.l7 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1, stride=1),
        )

        self.l8 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.in_ch, 3, padding=1, stride=1),
        )

    def forward(self, x, t):
        conv1 = self.l1(x)  # 124
        conv2 = self.l2(conv1+t)  # 120
        conv3 = self.l3(conv2+t)  # 116
        conv4 = self.l4(conv3+t)  # 112

        deconv5 = self.l5(conv4+t)  # 112
        deconv5 += conv3
        deconv5 = F.relu(deconv5)

        deconv6 = self.l6(deconv5+t)  # 116
        deconv6 += conv2
        deconv6 = F.relu(deconv6)

        deconv7 = self.l7(deconv6+t)
        deconv7 += conv1
        deconv7 = F.relu(deconv7)
        #
        deconv8 = self.l8(deconv7+t)
        deconv8 += x
        deconv8 = deconv8

        return deconv8

class IterBlock(nn.Module):
    def __init__(self):
        super(IterBlock, self).__init__()
        self.block1 = fidelity_module()

        self.relu = nn.ReLU()   
        self.red_tiny = REDCNN()

    def forward(self, input_data, proj, teb, t):
        tmp1 = self.block1(input_data, proj, t)
        tmp2 = self.red_tiny(input_data, teb)

        output = tmp1 + tmp2 
        output = self.relu(output)
        # output = output + input_data
        return output

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class LearnBlock(nn.Module):
    def __init__(self, block_num=1, **kwargs):
        super(LearnBlock, self).__init__()
        self.model = nn.ModuleList([IterBlock() for i in range(block_num)])

        inner_channel = 64
        self.time_emb = nn.Sequential(
                            TimeEmbedding(inner_channel),
                            nn.Linear(inner_channel, inner_channel * 4),
                            Swish(),
                            nn.Linear(inner_channel * 4, inner_channel)
                        )
    
    def forward(self, input_data, proj, t):
        x = input_data
        time_emb = self.time_emb(t)[:, :, None, None]
        for index, module in enumerate(self.model):
            xt = module(x, proj, time_emb, t)
            x = xt.clone()
        return x

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_alpha_schedule(timesteps):
    steps = timesteps
    alphas_cumprod = 1 - torch.linspace(0, steps, steps) / timesteps
    return torch.clip(alphas_cumprod, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        train_routine = 'Final',
        sampling_routine='default'
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.learn_block = LearnBlock()

        betas = linear_alpha_schedule(timesteps)
        # alphas = 1. - betas
        # alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod = betas

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1. - alphas_cumprod))

        self.train_routine = train_routine
        self.sampling_routine = sampling_routine

    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):

        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        noise = img
        xt = img
        x1_bar = img
        direct_recons = None
        sino = A(img, views)

        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            
            if t-2<0:
                step2 = torch.full((batch_size,), 0, dtype=torch.long, device=img.device)
            else:
                step2 = torch.full((batch_size,), t-2, dtype=torch.long, device=img.device)

            x1_bar = self.denoise_fn(img, step)
            x1_bar = self.learn_block(x1_bar, sino, step2)
            
            x2_bar = noise
            if direct_recons == None:
                direct_recons = x1_bar

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            img = img - xt_bar + xt_sub1_bar
            
            t = t - 1

        self.denoise_fn.train()

        return xt, direct_recons, img

    def q_sample(self, x_start, x_end, t):
        # simply use the alphas to interpolate
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )
    
    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    def p_losses(self, x_start, x_end, t):
        b, c, h, w = x_start.shape
        if self.train_routine == 'Final':
            x_mix = self.q_sample(x_start=x_start, x_end=x_end, t=t)
            x_recon = self.denoise_fn(x_mix, t)


            t_sub1 = t - 1
            t_sub1[t_sub1 < 0] = 0
            sino = A(x_end, views)
            x_recon_learn = self.learn_block(x_recon, sino, t_sub1)

            # sino = A(x_end, views)
            # x_recon_learn = self.learn_block(x_mix, sino, t)
            # x_recon = self.denoise_fn(x_recon_learn, t)

            if self.loss_type == 'l1':
                loss1 = (x_start - x_recon).abs().mean()
                # loss2 = (x_start - x_recon_learn).abs().mean()
                loss = loss1
            elif self.loss_type == 'l2':
                loss1 = F.mse_loss(x_start, x_recon)
                loss2 = F.mse_loss(x_start, x_recon_learn)
                print("loss1:{} loss2:{}".format(loss1, loss2))
                loss = loss1 + loss2
            else:
                raise NotImplementedError()

        return loss

    def forward(self, x1, x2, *args, **kwargs):
        b, c, h, w, device, img_size, = *x1.shape, x1.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x1, x2, t, *args, **kwargs)
    
