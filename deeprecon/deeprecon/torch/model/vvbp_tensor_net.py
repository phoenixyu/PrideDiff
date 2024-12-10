import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd
from torch.cuda.amp import custom_bwd
from .red_cnn import red_cnn
from .dncnn import dncnn
from .fbp_conv_net import fbp_conv_net
from ..utils.utils import create_projector2d

class bprj_sv_fun(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, projector, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = torch.zeros(input.shape[0], angles.shape[0], *input.shape[2:], device=input.device)
        for idx, angle in enumerate(angles):
            output[:, idx, None, ...] = projector.backprojection(input[:, :, idx, None, :], angle[None])
        self.projector = projector
        self.save_for_backward(angles)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        projector = self.projector
        temp = grad_output.sum(1, keepdim=True)
        grad_input = projector.backprojection_t(temp, *self.saved_tensors)
        return grad_input, None, None

class vvbp_tensor_net(nn.Module):
    def __init__(self, model_type="red_cnn", **kwargs):
        super().__init__()
        self.projector = create_projector2d(**kwargs)

        if "views" in kwargs and kwargs["views"] is not None:
            views = kwargs["views"]
        else:
            views = torch.arange(0, kwargs["num_views"]) * torch.pi * 2 / kwargs["num_views"]
        self.views = nn.parameter.Parameter(torch.FloatTensor(views), requires_grad=False)

        if model_type == "red_cnn":
            self.model = red_cnn(**kwargs)        
        elif model_type == "dncnn":
            self.model = dncnn(**kwargs)
        elif model_type == "fbp_conv_net":
            self.model = fbp_conv_net(**kwargs)
        else:
            raise NotImplementedError(f"Unexpected model type {model_type}")
        
        self.conv = nn.Conv2d(self.views.shape[0], 1, 3, padding=1)
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
        
    def forward(self, p):
        p_w = p * self.projector.weight
        filtered_p = nn.functional.conv2d(p_w, self.projector.filter, padding=(0, self.projector.num_det - 1))
        x = bprj_sv_fun.apply(filtered_p, self.projector, self.views)
        x_in, _ = torch.sort(x, dim=1)
        x_in = self.conv(x_in)
        out = self.model(x_in)
        return out
        
    def training_loss(self, p, y, **kwargs):
        out = self(p)
        loss = self.loss(out, y)
        return loss

    def test(self, p, **kwargs):
        return self(p)
