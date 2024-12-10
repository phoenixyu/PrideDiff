import torch.nn as nn
from torch import Tensor
from typing import List, Union, Tuple
from .function import *
from .filter import filter_generate


__functions__ = {
    "dis": {
        "flat":{
            "projection": ProjFanFlatDis2d,
            "projection_t": ProjTransFanFlatDis2d,
            "bprojection": BackProjFanFlatDis2d,
            "bprojection_t": BackProjTransFanFlatDis2d,
            "bprojection_w": BackProjWeightedFanFlatDis2d,
            "get_sys_matrix":GetSysMatrixFanFlat2d,

            "projection3d": ProjConeFlatDis3d,
            "projection3d_t": ProjTransConeFlatDis3d,
            "bprojection3d": BackProjConeFlatDis3d,
            "bprojection3d_t": BackProjTransConeFlatDis3d,
            "bprojection3d_w": BackProjWeightedConeFlatDis3d,
        },
        "arc": {
            "projection": ProjFanArcDis2d,
            "projection_t": ProjTransFanArcDis2d,
            "bprojection": BackProjFanArcDis2d,
            "bprojection_t": BackProjTransFanArcDis2d,
            "bprojection_w": BackProjWeightedFanArcDis2d,
            "get_sys_matrix":GetSysMatrixFanArc2d,

            "projection3d": ProjConeArcDis3d,
            "projection3d_t": ProjTransConeArcDis3d,
            "bprojection3d": BackProjConeArcDis3d,
            "bprojection3d_t": BackProjTransConeArcDis3d,
            "bprojection3d_w": BackProjWeightedConeArcDis3d,

            "projectionsp": ProjSpiralDis3d,
            "projectionsp_t": ProjTransSpiralDis3d,
            "bprojectionsp": BackProjSpiralDis3d,
            "bprojectionsp_t": BackProjTransSpiralDis3d,
            "bprojectionsp_w": BackProjWeightedSpiralDis3d,
        },
        "para": {
            "projection": ProjParaDis2d,
            "projection_t": ProjTransParaDis2d,
            "bprojection": BackProjParaDis2d,
            "bprojection_t": BackProjTransParaDis2d,
            "bprojection_w": BackProjParaDis2d,
            "get_sys_matrix":GetSysMatrixPara2d,
        }
    }
}

class projector2d(nn.Module):
    __doc__ = r"""2D projector for CT reconstruction.

    Args:
        image_size (int or two-tuple): Resolution of the image. If tuple, should be (width, height).
        num_det (int): Number of detector elements.
        pix_size (float or two-tuple): Physical size of each pixel. If tuple, should be (size_x, size_y).
        det_size (float): Physical length of each detector element.
        iso_source (float): The distance between x-ray source and iso center.
        source_detector (float): The distance between x-ray source and detector.
        pixshift (float or two-tuple, optional): The position offset of scanned object. Default: 0
        binshift (float, optional): The position offset of the detector. Default: 0
        scan_type (str, optional): The scanning mode, ``'flat'``, ``'arc'`` or ``'para'``. Default: ``'flat'``
        method (str, optional): The projection and backprojection method, ``'dis'``. Default: ``'dis'``
        filter_type (str, optional): The filter for FBP, ``'ramp'`` or ``'sheeplogan'``. Default: ``'ramp'``
        trainable (bool, optional): If ``True``, the filter and cosine weight are trainable. Default: ``False``

    Methods:
        projection(image: Tensor, angles: Tensor) -> Tensor: Projection
        backprojection(sino: Tensor, angles: Tensor) -> Tensor: Backprojection
        projection_t(sino: Tensor, angles: Tensor) -> Tensor: Transpose of projection
        backprojection_t(image: Tensor, angles: Tensor) -> Tensor: Transpose of backprojection
        filtered_backprojection(sino: Tensor, angles: Tensor, filter: Tensor | None = None, filter_type: str | None = None, redundant: bool = True) -> Tensor: FBP
    """
    def __init__(
        self,
        image_size: Union[int, Tuple[int], List[int]],
        num_det: int,
        pix_size: Union[float, Tuple[float], List[float]],
        det_size: float,
        iso_source: float,
        source_detector: float,
        pixshift: Union[float, Tuple[float], List[float]] = 0.,
        binshift: float = 0.,
        scan_type: str = "flat",
        method: str = "dis",
        filter_type: str = "ramp",
        trainable: bool = False,
        dtype = None
    ) -> None:
        super().__init__()
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size, image_size)
        assert len(image_size) == 2, "Image resolution should be a int or two-tuple int"
        if not isinstance(pix_size, (tuple, list)):
            pix_size = (pix_size, pix_size)
        assert len(pix_size) == 2, "Pixle size should be a float or two-tuple float"
        if not isinstance(pixshift, (tuple, list)):
            pixshift = (pixshift, pixshift)
        assert len(pixshift) == 2, "Scanned object shift should be a float or two-tuple float"
        scan_type = scan_type.lower()
        method = method.lower()
        filter_type = filter_type.lower()
        assert scan_type in {"flat", "arc", "para"}, f"Scanning mode {scan_type} not supported"
        assert method in {"dis"}, f"Radon transform method {method} not supported"
        if scan_type == "arc":
            det_size = det_size / source_detector
            binshift = binshift / source_detector
        options = torch.tensor([*image_size, num_det, *pix_size, det_size, iso_source, source_detector, *pixshift, binshift], dtype=dtype)
        self.image_size = tuple(image_size)
        self.num_det = num_det
        self.det_size = det_size
        self.binshift = binshift
        self.iso_source = iso_source
        self.source_detector = source_detector
        self.scan_type = scan_type
        self.method = method
        self.filter_type = filter_type
        self.options = nn.parameter.Parameter(options, requires_grad=False)
        self.filter = nn.parameter.Parameter(self.filter_generate(filter_type=self.filter_type, dtype=dtype), requires_grad=trainable)
        self.weight = nn.parameter.Parameter(self.weight_generate(dtype=dtype), requires_grad=trainable)

    def projection(self, image: Tensor, angles: Tensor) -> Tensor:
        b, c, h, w = image.shape
        assert (w, h) == self.image_size, "The size of input image is inconsistent with geometry"
        sino = __functions__[self.method][self.scan_type]["projection"].apply(image, self.options, angles)
        return sino

    def backprojection(self, sino: Tensor, angles: Tensor) -> Tensor:
        b, c, h, w = sino.shape
        v, = angles.shape
        assert (h, w) == (v, self.num_det), "The size of input sinogram is inconsistent with geometry"
        image = __functions__[self.method][self.scan_type]["bprojection"].apply(sino, self.options, angles)
        return image

    def backprojection_w(self, sino: Tensor, angles: Tensor) -> Tensor:
        b, c, h, w = sino.shape
        v, = angles.shape
        assert (h, w) == (v, self.num_det), "The size of input sinogram is inconsistent with geometry"
        image = __functions__[self.method][self.scan_type]["bprojection_w"].apply(sino, self.options, angles)
        return image

    def projection_t(self, sino: Tensor, angles: Tensor) -> Tensor:
        b, c, h, w = sino.shape
        v, = angles.shape
        assert (h, w) == (v, self.num_det), "The size of input sinogram is inconsistent with geometry"
        image = __functions__[self.method][self.scan_type]["projection_t"].apply(sino, self.options, angles)
        return image

    def backprojection_t(self, image: Tensor, angles: Tensor) -> Tensor:
        b, c, h, w = image.shape
        assert (w, h) == self.image_size, "The size of input image is inconsistent with geometry"
        sino = __functions__[self.method][self.scan_type]["bprojection_t"].apply(image, self.options, angles)
        return sino

    def forward(self, image: Tensor, angles: Tensor) -> Tensor:
        return self.projection(image, angles)

    def filter_generate(self, filter_type, dtype=None):
        iso_source = self.iso_source
        source_detector = self.source_detector
        if self.scan_type == "flat":
            det_size = self.det_size * iso_source / source_detector
        else:
            det_size = self.det_size
        num_det = self.num_det
        filter = filter_generate(num_det, det_size, filter_type, dtype)
        filter = filter[None, None, None, :]
        return filter

    def weight_generate(self, dtype=None):
        det_size = self.det_size
        num_det = self.num_det
        binshift = self.binshift
        iso_source = self.iso_source
        source_detector = self.source_detector
        weight = torch.arange((-num_det / 2 + 0.5), num_det / 2, 1, dtype=dtype)
        weight = weight * det_size + binshift
        if self.scan_type == "flat":
            weight = weight * iso_source / source_detector
            weight = iso_source / torch.sqrt(weight ** 2 + iso_source ** 2)
        elif self.scan_type == "arc":
            weight = torch.cos(weight)
        elif self.scan_type == "para":
            weight = torch.ones_like(weight)
        return weight

    def filtered_backprojection(
        self, sino: Tensor, 
        angles: Tensor, 
        filter: Union[Tensor, None] = None, 
        filter_type: Union[str, None] = None, 
        redundant: bool = True
    ) -> Tensor:
        
        if filter is not None:
            _filter = filter
        else:             
            if filter_type is None or filter_type.lower() == self.filter_type:
                _filter = self.filter
            else:
                _filter = self.filter_generate(filter_type=filter_type.lower(), dtype=sino.dtype).to(sino.device)
        sino = sino * self.weight
        filtered_sino = nn.functional.conv2d(sino, _filter, padding=(0, self.num_det - 1))
        recon = self.backprojection_w(filtered_sino, angles)
        recon = recon * (angles[-1] - angles[0]).abs() / (angles.size(0) - 1)
        if redundant:
            recon = recon / 2
        return recon
    
    def get_sys_matrix(self, angles: Tensor) -> Tensor:
        return __functions__[self.method][self.scan_type]["get_sys_matrix"](self.options, angles)


class projector3d(nn.Module):
    __doc__ = r"""3D projector for CT reconstruction.

    Args:
        image_size (int or three-tuple): Resolution of the image. If tuple, should be (width, height, depth).
        num_det (int or two-tuple): Number of detector elements. If tuple, should be (col, row).
        pix_size (float or three-tuple): Physical size of each pixel. If tuple, should be (size_x, size_y, size_z).
        det_size (float or two-tuple): Physical length of each detector element. If tuple, should be (size_col, size_row).
        iso_source (float): The distance between x-ray source and iso center.
        source_detector (float): The distance between x-ray source and detector.
        pixshift (float or three-tuple, optional): The position offset of scanned object. Default: 0
        binshift (float or two-tuple, optional): The position offset of the detector. Default: 0
        scan_type (str, optional): The scanning mode, ``'flat'`` or ``'arc'``. Default: ``'flat'``
        method (str, optional): The projection and backprojection method, ``'dis'``. Default: ``'dis'``
        filter_type (str, optional): The filter for FBP, ``'ramp'`` or ``'sheeplogan'``. Default: ``'ramp'``
        trainable (bool, optional): If ``True``, the filter and cosine weight are trainable. Default: ``False``

    Methods:
        projection(image: Tensor, angles: Tensor) -> Tensor: Projection
        backprojection(sino: Tensor, angles: Tensor) -> Tensor: Backprojection
        projection_t(sino: Tensor, angles: Tensor) -> Tensor: Transpose of projection
        backprojection_t(image: Tensor, angles: Tensor) -> Tensor: Transpose of backprojection
        filtered_backprojection(sino: Tensor, angles: Tensor, filter: Tensor | None = None, filter_type: str | None = None, redundant: bool = True) -> Tensor: FBP
    """
    def __init__(
        self,
        image_size: Union[int, Tuple[int], List[int]],
        num_det: Union[int, Tuple[int], List[int]],
        pix_size: Union[float, Tuple[float], List[float]],
        det_size: Union[float, Tuple[float], List[float]],
        iso_source: float,
        source_detector: float,
        pixshift: Union[float, Tuple[float], List[float]] = 0.,
        binshift: Union[float, Tuple[float], List[float]] = 0.,
        scan_type: str = "flat",
        method: str = "dis",
        filter_type: str = "ramp",
        trainable: bool = False,
        dtype = None
    ) -> None:
        super().__init__()
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size, image_size, image_size)
        assert len(image_size) == 3, "Image resolution should be a int or three-tuple int"
        if not isinstance(num_det, (tuple, list)):
            num_det = (num_det, num_det)
        assert len(num_det) == 2 , "Number of detector elements should be a int or two-tuple int"
        if not isinstance(pix_size, (tuple, list)):
            pix_size = (pix_size, pix_size, pix_size)
        assert len(pix_size) == 3, "Pixle size should be a float or three-tuple float"
        if not isinstance(det_size, (tuple, list)):
            det_size = (det_size, det_size)
        assert len(det_size) == 2, "The size of detector element should be a float or two-tuple float"
        if not isinstance(pixshift, (tuple, list)):
            pixshift = (pixshift, pixshift, pixshift)
        assert len(pixshift) == 3, "Scanned object shift should be a float or three-tuple float"
        if not isinstance(binshift, (tuple, list)):
            binshift = (binshift, binshift)
        assert len(binshift) == 2, "Detector shift should be a float or two-tuple float"
        scan_type = scan_type.lower()
        method = method.lower()
        filter_type = filter_type.lower()
        assert scan_type in {"flat", "arc"}, f"Scanning mode {scan_type} not supported"
        assert method in {"dis"}, f"Radon transform method {method} not supported"
        if scan_type == "arc":
            det_size = (det_size[0] / source_detector, det_size[1])
            binshift = (binshift[0] / source_detector, binshift[1])
        options = torch.tensor([*image_size, *num_det, *pix_size, *det_size, iso_source, source_detector, *pixshift, *binshift], dtype=dtype)
        self.image_size = tuple(image_size)
        self.num_det = tuple(num_det)
        self.det_size = det_size
        self.binshift = binshift
        self.iso_source = iso_source
        self.source_detector = source_detector
        self.scan_type = scan_type
        self.method = method
        self.filter_type = filter_type
        self.options = nn.parameter.Parameter(options, requires_grad=False)
        self.filter = nn.parameter.Parameter(self.filter_generate(filter_type=self.filter_type, dtype=dtype), requires_grad=trainable)
        self.weight = nn.parameter.Parameter(self.weight_generate(dtype=dtype), requires_grad=trainable)

    def projection(self, image: Tensor, angles: Tensor) -> Tensor:
        b, c, d, h, w = image.shape
        assert (w, h, d) == self.image_size, "The size of input image is inconsistent with geometry"
        sino = __functions__[self.method][self.scan_type]["projection3d"].apply(image, self.options, angles)
        return sino

    def backprojection(self, sino: Tensor, angles: Tensor) -> Tensor:
        b, c, d, h, w = sino.shape
        v, = angles.shape
        assert (d, w, h) == (v, *self.num_det), "The size of input sinogram is inconsistent with geometry"
        image = __functions__[self.method][self.scan_type]["bprojection3d"].apply(sino, self.options, angles)
        return image

    def backprojection_w(self, sino: Tensor, angles: Tensor) -> Tensor:
        b, c, d, h, w = sino.shape
        v, = angles.shape
        assert (d, w, h) == (v, *self.num_det), "The size of input sinogram is inconsistent with geometry"
        image = __functions__[self.method][self.scan_type]["bprojection3d_w"].apply(sino, self.options, angles)
        return image

    def projection_t(self, sino: Tensor, angles: Tensor) -> Tensor:
        b, c, d, h, w = sino.shape
        v, = angles.shape
        assert (d, w, h) == (v, *self.num_det), "The size of input sinogram is inconsistent with geometry"
        image = __functions__[self.method][self.scan_type]["projection3d_t"].apply(sino, self.options, angles)
        return image

    def backprojection_t(self, image: Tensor, angles: Tensor) -> Tensor:
        b, c, d, h, w = image.shape
        assert (w, h, d) == self.image_size, "The size of input image is inconsistent with geometry"
        sino = __functions__[self.method][self.scan_type]["bprojection3d_t"].apply(image, self.options, angles)
        return sino

    def forward(self, image: Tensor, angles: Tensor) -> Tensor:
        return self.projection(image, angles)

    def filter_generate(self, filter_type, dtype=None):
        iso_source = self.iso_source
        source_detector = self.source_detector
        if self.scan_type == "flat":
            det_size = self.det_size[0] * iso_source / source_detector
        else:
            det_size = self.det_size[0]
        num_det = self.num_det[0]
        filter = filter_generate(num_det, det_size, filter_type, dtype)
        filter = filter[None, None, None, None, :]
        return filter

    def weight_generate(self, dtype=None):
        col_size, row_size = self.det_size
        col_n, row_n = self.num_det
        col_s, row_s = self.binshift
        iso_source = self.iso_source
        source_detector = self.source_detector
        weight_row = torch.arange((-row_n / 2 + 0.5), row_n / 2, 1, dtype=dtype)
        weight_row = weight_row * row_size + row_s
        weight_row = weight_row * iso_source / source_detector
        weight_col = torch.arange((-col_n / 2 + 0.5), col_n / 2, 1, dtype=dtype)
        weight_col = weight_col * col_size + col_s
        if self.scan_type == "flat":
            weight_col = weight_col * iso_source / source_detector
        elif self.scan_type == "arc":
            weight_col = torch.tan(weight_col) * iso_source
        weight_row = weight_row[:, None].repeat(1, col_n)
        weight_col = weight_col[None, :].repeat(row_n, 1)
        weight = iso_source / torch.sqrt(weight_row ** 2 + weight_col ** 2 + iso_source ** 2)
        return weight

    def filtered_backprojection(
        self, 
        sino: Tensor, 
        angles: Tensor, 
        filter: Union[Tensor, None] = None, 
        filter_type: Union[str, None] = None, 
        redundant: bool = True
    ) -> Tensor:
        
        if filter is not None:
            _filter = filter
        else:             
            if filter_type is None or filter_type.lower() == self.filter_type:
                _filter = self.filter
            else:
                _filter = self.filter_generate(filter_type=filter_type.lower(), dtype=sino.dtype).to(sino.device)
        sino = sino * self.weight
        filtered_sino = nn.functional.conv3d(sino, _filter, padding=(0, 0, self.num_det[0] - 1))
        recon = self.backprojection_w(filtered_sino, angles)
        recon = recon * (angles[-1] - angles[0]).abs() / (angles.size(0) - 1)
        if redundant:
            recon = recon / 2
        return recon


class projectorsp(nn.Module):
    __doc__ = r"""Spiral projector for CT reconstruction.

    Args:
        image_size (int or three-tuple): Resolution of the image. If tuple, should be (width, height, depth).
        num_det (int or two-tuple): Number of detector elements. If tuple, should be (col, row).
        pix_size (float or three-tuple): Physical size of each pixel. If tuple, should be (size_x, size_y, size_z).
        det_size (float or two-tuple): Physical length of each detector element. If tuple, should be (size_col, size_row).
        iso_source (float): The distance between x-ray source and iso center.
        source_detector (float): The distance between x-ray source and detector.
        pixshift (float or two-tuple, optional): The position offset of scanned object. Default: 0
        binshift (float or two-tuple, optional): The position offset of the detector. Default: 0
        scan_type (str, optional): The scanning mode, ``'flat'``. Default: ``'flat'``
        method (str, optional): The projection and backprojection method, ``'dis'``. Default: ``'dis'``
        filter_type (str, optional): The filter for FBP, ``'ramp'`` or ``'sheeplogan'``. Default: ``'ramp'``
        trainable (bool, optional): If ``True``, the filter and cosine weight are trainable. Default: ``False``

    Methods:
        projection(image: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor: Projection
        backprojection(sino: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor: Backprojection
        projection_t(sino: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor: Transpose of projection
        backprojection_t(image: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor: Transpose of backprojection
        filtered_backprojection(sino: Tensor, angles: Tensor, , source_pos: Tensor, filter_type: str | None = None) -> Tensor: FBP
    """
    def __init__(
        self,
        image_size: Union[int, Tuple[int], List[int]],
        num_det: Union[int, Tuple[int], List[int]],
        pix_size: Union[float, Tuple[float], List[float]],
        det_size: Union[float, Tuple[float], List[float]],
        iso_source: float,
        source_detector: float,
        pixshift: Union[float, Tuple[float], List[float]] = 0.,
        binshift: Union[float, Tuple[float], List[float]] = 0.,
        scan_type: str = "arc",
        method: str = "dis",
        filter_type: str = "ramp",
        trainable: bool = False,
        dtype = None
    ) -> None:
        super().__init__()
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size, image_size, image_size)
        assert len(image_size) == 3, "Image resolution should be a int or three-tuple int"
        if not isinstance(num_det, (tuple, list)):
            num_det = (num_det, num_det)
        assert len(num_det) == 2 , "Number of detector elements should be a int or two-tuple int"
        if not isinstance(pix_size, (tuple, list)):
            pix_size = (pix_size, pix_size, pix_size)
        assert len(pix_size) == 3, "Pixle size should be a float or three-tuple float"
        if not isinstance(det_size, (tuple, list)):
            det_size = (det_size, det_size)
        assert len(det_size) == 2, "The size of detector element should be a float or two-tuple float"
        if not isinstance(pixshift, (tuple, list)):
            pixshift = (pixshift, pixshift)
        assert len(pixshift) == 2, "Scanned object shift should be a float or two-tuple float"
        if not isinstance(binshift, (tuple, list)):
            binshift = (binshift, binshift)
        assert len(binshift) == 2, "Detector shift should be a float or two-tuple float"
        scan_type = scan_type.lower()
        method = method.lower()
        filter_type = filter_type.lower()
        assert scan_type in {"arc"}, f"Scanning mode {scan_type} not supported"
        assert method in {"dis"}, f"Radon transform method {method} not supported"
        if scan_type == "arc":
            det_size = (det_size[0] / source_detector, det_size[1])
            binshift = (binshift[0] / source_detector, binshift[1])
        options = torch.tensor([*image_size, *num_det, *pix_size, *det_size, iso_source, source_detector, *pixshift, *binshift], dtype=dtype)
        self.image_size = tuple(image_size)
        self.num_det = tuple(num_det)
        self.det_size = det_size
        self.binshift = binshift
        self.iso_source = iso_source
        self.source_detector = source_detector
        self.scan_type = scan_type
        self.method = method
        self.filter_type = filter_type
        self.options = nn.parameter.Parameter(options, requires_grad=False)
        self.filter = nn.parameter.Parameter(self.filter_generate(filter_type=self.filter_type, dtype=dtype), requires_grad=trainable)
        self.weight = nn.parameter.Parameter(self.weight_generate(dtype=dtype), requires_grad=trainable)

    def projection(self, image: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor:
        b, c, d, h, w = image.shape
        v, = angles.shape
        m, n = source_pos.shape
        assert (w, h, d) == self.image_size, "The size of input image is inconsistent with geometry"
        assert (m, n) == (4, v), "The size of angles and source position is not consistent"
        sino = __functions__[self.method][self.scan_type]["projectionsp"].apply(image, self.options, angles, source_pos)
        return sino

    def backprojection(self, sino: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor:
        b, c, d, h, w = sino.shape
        v, = angles.shape
        m, n = source_pos.shape
        assert (d, w, h) == (v, *self.num_det), "The size of input sinogram is inconsistent with geometry"
        assert (m, n) == (4, v), "The size of angles and source position is not consistent"
        image = __functions__[self.method][self.scan_type]["bprojectionsp"].apply(sino, self.options, angles, source_pos)
        return image

    def backprojection_w(self, sino: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor:
        b, c, d, h, w = sino.shape
        v, = angles.shape
        m, n = source_pos.shape
        assert (d, w, h) == (v, *self.num_det), "The size of input sinogram is inconsistent with geometry"
        assert (m, n) == (4, v), "The size of angles and source position is not consistent"
        image = __functions__[self.method][self.scan_type]["bprojectionsp_w"].apply(sino, self.options, angles, source_pos)
        return image

    def projection_t(self, sino: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor:
        b, c, d, h, w = sino.shape
        v, = angles.shape
        m, n = source_pos.shape
        assert (d, w, h) == (v, *self.num_det), "The size of input sinogram is inconsistent with geometry"
        assert (m, n) == (4, v), "The size of angles and source position is not consistent"
        image = __functions__[self.method][self.scan_type]["projectionsp_t"].apply(sino, self.options, angles, source_pos)
        return image

    def backprojection_t(self, image: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor:
        b, c, d, h, w = image.shape
        v, = angles.shape
        m, n = source_pos.shape
        assert (w, h, d) == self.image_size, "The size of input image is inconsistent with geometry"
        assert (m, n) == (4, v), "The size of angles and source position is not consistent"
        sino = __functions__[self.method][self.scan_type]["bprojectionsp_t"].apply(image, self.options, angles, source_pos)
        return sino

    def forward(self, image: Tensor, angles: Tensor, source_pos: Tensor) -> Tensor:
        return self.projection(image, angles, source_pos)

    def filter_generate(self, filter_type, dtype=None):
        iso_source = self.iso_source
        source_detector = self.source_detector
        if self.scan_type == "flat":
            det_size = self.det_size[0] * iso_source / source_detector
        else:
            det_size = self.det_size[0]
        num_det = self.num_det[0]
        filter = filter_generate(num_det, det_size, filter_type, dtype)
        filter = filter[None, None, None, None, :]
        return filter

    def weight_generate(self, dtype=None):
        col_size, row_size = self.det_size
        col_n, row_n = self.num_det
        col_s, row_s = self.binshift
        iso_source = self.iso_source
        source_detector = self.source_detector
        weight_row = torch.arange((-row_n / 2 + 0.5), row_n / 2, 1, dtype=dtype)
        weight_row = weight_row * row_size + row_s
        weight_row = weight_row * iso_source / source_detector
        weight_col = torch.arange((-col_n / 2 + 0.5), col_n / 2, 1, dtype=dtype)
        weight_col = weight_col * col_size + col_s
        if self.scan_type == "flat":
            weight_col = weight_col * iso_source / source_detector
        elif self.scan_type == "arc":
            weight_col = torch.tan(weight_col) * iso_source
        weight_row = weight_row[:, None].repeat(1, col_n)
        weight_col = weight_col[None, :].repeat(row_n, 1)
        weight = iso_source / torch.sqrt(weight_row ** 2 + weight_col ** 2 + iso_source ** 2)
        return weight

    def filtered_backprojection(self, sino: Tensor, angles: Tensor, source_pos: Tensor, filter_type: Union[str, None] = None) -> Tensor:
        if filter_type is None or filter_type.lower() == self.filter_type:
            filter = self.filter
        else:
            filter = self.filter_generate(filter_type=filter_type.lower(), dtype=sino.dtype).to(sino.device)
        sino = sino * self.weight
        filtered_sino = nn.functional.conv3d(sino, filter, padding=(0, 0, self.num_det[0] - 1))
        recon = self.backprojection_w(filtered_sino, angles, source_pos)
        recon = recon * abs(angles[1] - angles[0]) / 2
        return recon