import torch
import radontorch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd
from torch.cuda.amp import custom_bwd

class ProjFanFlatDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjFanFlatDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjTransFanFlatDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class ProjTransFanFlatDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjTransFanFlatDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjFanFlatDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjFanFlatDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjFanFlatDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransFanFlatDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjTransFanFlatDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjTransFanFlatDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjFanFlatDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjWeightedFanFlatDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjWeightedFanFlatDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransWeightedFanFlatDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None


class ProjFanArcDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjFanArcDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjTransFanArcDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class ProjTransFanArcDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjTransFanArcDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjFanArcDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjFanArcDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjFanArcDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransFanArcDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjTransFanArcDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjTransFanArcDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjFanArcDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjWeightedFanArcDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjWeightedFanArcDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransWeightedFanArcDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None


class ProjParaDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjParaDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjTransParaDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class ProjTransParaDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjTransParaDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjParaDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjParaDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjParaDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransParaDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjTransParaDis2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjTransParaDis2d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjParaDis2d(grad_output, *self.saved_tensors)
        return grad_input, None, None


class ProjConeFlatDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjConeFlatDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjTransConeFlatDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class ProjTransConeFlatDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjTransConeFlatDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjConeFlatDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjConeFlatDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjConeFlatDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransConeFlatDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjTransConeFlatDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjTransConeFlatDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjConeFlatDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjWeightedConeFlatDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjWeightedConeFlatDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransWeightedConeFlatDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None


class ProjConeArcDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjConeArcDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjTransConeArcDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class ProjTransConeArcDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjTransConeArcDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjConeArcDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjConeArcDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjConeArcDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransConeArcDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjTransConeArcDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjTransConeArcDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjConeArcDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None

class BackProjWeightedConeArcDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjWeightedConeArcDis3d(input, option, angles)
        self.save_for_backward(option, angles)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransWeightedConeArcDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None


class ProjSpiralDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles, source_pos):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjSpiralDis3d(input, option, angles, source_pos)
        self.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjTransSpiralDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None, None

class ProjTransSpiralDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles, source_pos):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.ProjTransSpiralDis3d(input, option, angles, source_pos)
        self.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.ProjSpiralDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None, None

class BackProjSpiralDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles, source_pos):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjSpiralDis3d(input, option, angles, source_pos)
        self.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransSpiralDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None, None

class BackProjTransSpiralDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles, source_pos):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjTransSpiralDis3d(input, option, angles, source_pos)
        self.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjSpiralDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None, None

class BackProjWeightedSpiralDis3d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, option, angles, source_pos):
        if not input.is_contiguous():
            input = input.contiguous()
        output = radontorch.BackProjWeightedSpiralDis3d(input, option, angles, source_pos)
        self.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    @custom_bwd
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = radontorch.BackProjTransWeightedSpiralDis3d(grad_output, *self.saved_tensors)
        return grad_input, None, None, None


def GetSysMatrixFanFlat2d(option, angles):
    option = option.to("cpu").to(torch.float32)
    angles = angles.to("cpu").to(torch.float32)
    matrix = radontorch.GetSysMatrixFanFlat2d(option, angles)
    return matrix


def GetSysMatrixFanArc2d(option, angles):
    option = option.to("cpu").to(torch.float32)
    angles = angles.to("cpu").to(torch.float32)
    matrix = radontorch.GetSysMatrixFanArc2d(option, angles)
    return matrix


def GetSysMatrixPara2d(option, angles):
    option = option.to("cpu").to(torch.float32)
    angles = angles.to("cpu").to(torch.float32)
    matrix = radontorch.GetSysMatrixPara2d(option, angles)
    return matrix