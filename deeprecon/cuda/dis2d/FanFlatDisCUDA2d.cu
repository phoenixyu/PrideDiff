#include <cuda.h>

#define BLOCK_DIM 256


namespace
{
__device__ float Map2x(float sourcex, float sourcey, float pointx, float pointy) {
    return (sourcex * pointy - sourcey * pointx) / (pointy - sourcey);
}

__device__ float Map2y(float sourcex, float sourcey, float pointx, float pointy) {
    return (sourcey * pointx - sourcex * pointy) / (pointx - sourcex);
}

__device__ float CoordinateWeight(float sourcex, float sourcey, float pointx, float pointy, float r) {
    float d = (sourcex * pointx + sourcey * pointy) / r;
    return r * r / ((r - d) * (r - d));
}

__device__ float TriAngCos(float a, float b) {
    return abs(b) / sqrt(a * a + b * b);
}
}


__global__ void ProjFanFlatDisCUDA2dKernel(
    float* __restrict__ Projection,
    cudaTextureObject_t texObj,
    const float* __restrict__ ViewAngle,
    const int Width,
    const int Height,
    const int NumView,
    const int NumDet,
    const float ImageSizeX,
    const float ImageSizeY,
    const float DetSize,
    const float IsoSource,
    const float SourceDetector,
    const float PixXShift,
    const float PixYShift,
    const float BinShift) {

    const int idxBatch = blockIdx.x ;
    const int idxView = blockIdx.y;
    const int idxDet0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idxDet = idxDet0 + tx;
    const int MaxNDet = ((idxDet0 + blockDim.x) > NumDet) ? (NumDet - idxDet0) : blockDim.x;
    __shared__ float ProjTemp[BLOCK_DIM];
    __shared__ float DetProj2Axis[BLOCK_DIM + 1];

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float sourcex = - sinVal * IsoSource;
    float sourcey = cosVal * IsoSource;
    float VirDetSize = IsoSource / SourceDetector * DetSize;
    float VirBinShift = IsoSource / SourceDetector * BinShift;
    float Det0x;
    float Det0y;
    float Det1x;
    float Det1y;
    if (cosVal * cosVal > 0.5) {
        ProjTemp[tx] = 0;
        if (cosVal >= 0) {
            Det0x = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
        } else {
            Det0x = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * sinVal;
        }
        float Det0Proj = Map2x(sourcex, sourcey, Det0x, Det0y);
        float Det1Proj = Map2x(sourcex, sourcey, Det1x, Det1y);
        if (tx == 0) DetProj2Axis[tx] = Det0Proj;
        DetProj2Axis[tx + 1] = Det1Proj;
        __syncthreads();
        float coef1 = Det1Proj - DetProj2Axis[tx];
        float coef2 = TriAngCos((Det1Proj + DetProj2Axis[tx]) / 2.0 - sourcex, sourcey);
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
        float DetInterval = VirDetSize * abs(cosVal);
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float Bound0 = max(Point0Proj, DetProj2Axis[0]);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxd = floor((Bound0Proj2Det - Det0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                float temp = 0;
                while(idxcol < Width && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        temp += (Point1Proj - Bound0) * tex3D<float>(texObj, idxcol, idxrow, idxBatch);
                        Bound0 = Point1Proj;
                        idxcol++;
                        Point1Proj += PixInterval;
                    } else {
                        temp += (Det1Proj - Bound0) * tex3D<float>(texObj, idxcol, idxrow, idxBatch);
                        atomicAdd(ProjTemp + idxd, temp);
                        temp = 0;
                        Bound0 = Det1Proj;
                        idxd ++;
                        if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(ProjTemp + idxd, temp);
            }
        }
        __syncthreads();
        if (idxDet < NumDet) {
            ProjTemp[tx] *= ImageSizeY / (coef1 * coef2);
            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] = ProjTemp[tx];
        }
    } else {
        ProjTemp[tx] = 0;
        if (sinVal >= 0) {
            Det0x = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
        } else {
            Det0x = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * sinVal;
        }
        float Det0Proj = Map2y(sourcex, sourcey, Det0x, Det0y);
        float Det1Proj = Map2y(sourcex, sourcey, Det1x, Det1y);
        if (tx == 0) DetProj2Axis[tx] = Det0Proj;
        DetProj2Axis[tx + 1] = Det1Proj;
        __syncthreads();
        float coef1 = Det1Proj - DetProj2Axis[tx];
        float coef2 = TriAngCos((Det1Proj + DetProj2Axis[tx]) / 2.0 - sourcey, sourcex);
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
        float DetInterval = VirDetSize * abs(sinVal);
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float Bound0 = max(Point0Proj, DetProj2Axis[0]);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxd = floor((Bound0Proj2Det - Det0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                float temp = 0;
                while(idxrow < Height && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        temp += (Point1Proj - Bound0) * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch);
                        Bound0 = Point1Proj;
                        idxrow++;
                        Point1Proj += PixInterval;
                    } else {
                        temp += (Det1Proj - Bound0) * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch);
                        atomicAdd(ProjTemp + idxd, temp);
                        temp = 0;
                        Bound0 = Det1Proj;
                        idxd ++;
                        if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(ProjTemp + idxd, temp);
            }
        }
        __syncthreads();
        if (idxDet < NumDet) {
            ProjTemp[tx] *= ImageSizeX / (coef1 * coef2);
            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] = ProjTemp[tx];
        }
    }
}


__global__ void ProjTransFanFlatDisCUDA2dKernel(
    float* __restrict__ Image,
    const float* __restrict__ Projection,
    const float* __restrict__ ViewAngle,
    const int Width,
    const int Height,
    const int NumView,
    const int NumDet,
    const float ImageSizeX,
    const float ImageSizeY,
    const float DetSize,
    const float IsoSource,
    const float SourceDetector,
    const float PixXShift,
    const float PixYShift,
    const float BinShift) {

    const int idxBatch = blockIdx.x;
    const int idxView = blockIdx.y;
    const int idxDet0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idxDet = idxDet0 + tx;
    const int MaxNDet = ((idxDet0 + blockDim.x) > NumDet) ? (NumDet - idxDet0) : blockDim.x;
    __shared__ float ProjTemp[BLOCK_DIM];
    __shared__ float DetProj2Axis[BLOCK_DIM + 1];

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float sourcex = - sinVal * IsoSource;
    float sourcey = cosVal * IsoSource;
    float VirDetSize = IsoSource / SourceDetector * DetSize;
    float VirBinShift = IsoSource / SourceDetector * BinShift;
    float Det0x;
    float Det0y;
    float Det1x;
    float Det1y;
    if (cosVal * cosVal > 0.5) {
        if (idxDet < NumDet) {
            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet-1-idxDet);
            ProjTemp[tx] = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
        } else {
            ProjTemp[tx] = 0;
        }
        if (cosVal >= 0) {
            Det0x = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
        } else {
            Det0x = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * sinVal;
        }
        float Det0Proj = Map2x(sourcex, sourcey, Det0x, Det0y);
        float Det1Proj = Map2x(sourcex, sourcey, Det1x, Det1y);
        if (tx == 0) DetProj2Axis[tx] = Det0Proj;
        DetProj2Axis[tx + 1] = Det1Proj;
        __syncthreads();
        float coef1 = Det1Proj - DetProj2Axis[tx];
        float coef2 = TriAngCos((Det1Proj + DetProj2Axis[tx]) / 2.0 - sourcex, sourcey);
        ProjTemp[tx] *= ImageSizeY / (coef1 * coef2);
        __syncthreads();
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
        float DetInterval = VirDetSize * abs(cosVal);
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float Bound0 = max(Point0Proj, DetProj2Axis[0]);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxd = floor((Bound0Proj2Det - Det0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                float temp = 0;
                while(idxcol < Width && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        temp += (Point1Proj - Bound0) * ProjTemp[idxd];
                        atomicAdd(Image + (idxBatch * Height + idxrow) * Width + idxcol, temp);
                        temp = 0;
                        Bound0 = Point1Proj;
                        idxcol++;
                        Point1Proj += PixInterval;
                    } else {
                        temp += (Det1Proj - Bound0) * ProjTemp[idxd];
                        Bound0 = Det1Proj;
                        idxd ++;
                        if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(Image + (idxBatch * Height + idxrow) * Width + idxcol, temp);
            }
        }
    } else {
        if (idxDet < NumDet) {
            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet-1-idxDet);
            ProjTemp[tx] = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
        } else {
            ProjTemp[tx] = 0;
        }
        if (sinVal >= 0) {
            Det0x = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
        } else {
            Det0x = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * sinVal;
        }
        float Det0Proj = Map2y(sourcex, sourcey, Det0x, Det0y);
        float Det1Proj = Map2y(sourcex, sourcey, Det1x, Det1y);
        if (tx == 0) DetProj2Axis[tx] = Det0Proj;
        DetProj2Axis[tx + 1] = Det1Proj;
        __syncthreads();
        float coef1 = Det1Proj - DetProj2Axis[tx];
        float coef2 = TriAngCos((Det1Proj + DetProj2Axis[tx]) / 2.0 - sourcey, sourcex);
        ProjTemp[tx] *= ImageSizeX / (coef1 * coef2);
        __syncthreads();
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
        float DetInterval = VirDetSize * abs(sinVal);
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float Bound0 = max(Point0Proj, DetProj2Axis[0]);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxd = floor((Bound0Proj2Det - Det0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                float temp = 0;
                while(idxrow < Height && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        temp += (Point1Proj - Bound0) * ProjTemp[idxd];
                        atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, temp);
                        temp = 0;
                        Bound0 = Point1Proj;
                        idxrow++;
                        Point1Proj += PixInterval;
                    } else {
                        temp += (Det1Proj - Bound0) * ProjTemp[idxd];
                        Bound0 = Det1Proj;
                        idxd ++;
                        if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, temp);
            }
        }
    }
}


template<bool FBPWEIGHT>
__global__ void BackProjFanFlatDisCUDA2dKernel(
    float* __restrict__ Image,
    const float* __restrict__ Projection,
    const float* __restrict__ ViewAngle,
    const int Width,
    const int Height,
    const int NumView,
    const int NumDet,
    const float ImageSizeX,
    const float ImageSizeY,
    const float DetSize,
    const float IsoSource,
    const float SourceDetector,
    const float PixXShift,
    const float PixYShift,
    const float BinShift) {

    const int idxBatch = blockIdx.x;
    const int idxView = blockIdx.y;
    const int idxDet0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idxDet = idxDet0 + tx;
    const int MaxNDet = ((idxDet0 + blockDim.x) > NumDet) ? (NumDet - idxDet0) : blockDim.x;
    __shared__ float ProjTemp[BLOCK_DIM];
    __shared__ float DetProj2Axis[BLOCK_DIM + 1];

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float sourcex = - sinVal * IsoSource;
    float sourcey = cosVal * IsoSource;
    float VirDetSize = IsoSource / SourceDetector * DetSize;
    float VirBinShift = IsoSource / SourceDetector * BinShift;
    float Det0x;
    float Det0y;
    float Det1x;
    float Det1y;
    if (cosVal * cosVal > 0.5) {
        if (idxDet < NumDet) {
            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet-1-idxDet);
            ProjTemp[tx] = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] * VirDetSize;
        } else {
            ProjTemp[tx] = 0;
        }
        if (cosVal >= 0) {
            Det0x = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
        } else {
            Det0x = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * sinVal;
        }
        float Det0Proj = Map2x(sourcex, sourcey, Det0x, Det0y);
        float Det1Proj = Map2x(sourcex, sourcey, Det1x, Det1y);
        if (tx == 0) DetProj2Axis[tx] = Det0Proj;
        DetProj2Axis[tx + 1] = Det1Proj;
        __syncthreads();
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
        float DetInterval = VirDetSize * abs(cosVal);
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float Bound0 = max(Point0Proj, DetProj2Axis[0]);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxd = floor((Bound0Proj2Det - Det0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Pointx = (idxcol + 0.5) * ImageSizeX + Point0x;
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                float temp = 0;
                while(idxcol < Width && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        temp += (Point1Proj - Bound0) * ProjTemp[idxd];
                        if (FBPWEIGHT) temp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        temp /= PixInterval;
                        atomicAdd(Image + (idxBatch * Height + idxrow) * Width + idxcol, temp);
                        temp = 0;
                        Bound0 = Point1Proj;
                        idxcol++;
                        Pointx += ImageSizeX;
                        Point1Proj += PixInterval;
                    } else {
                        temp += (Det1Proj - Bound0) * ProjTemp[idxd];
                        Bound0 = Det1Proj;
                        idxd ++;
                        if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) {
                    if (FBPWEIGHT) temp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                    temp /= PixInterval;
                    atomicAdd(Image + (idxBatch * Height + idxrow) * Width + idxcol, temp);
                }
            }
        }
    } else {
        if (idxDet < NumDet) {
            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet-1-idxDet);
            ProjTemp[tx] = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] * VirDetSize;
        } else {
            ProjTemp[tx] = 0;
        }
        if (sinVal >= 0) {
            Det0x = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
        } else {
            Det0x = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * sinVal;
        }
        float Det0Proj = Map2y(sourcex, sourcey, Det0x, Det0y);
        float Det1Proj = Map2y(sourcex, sourcey, Det1x, Det1y);
        if (tx == 0) DetProj2Axis[tx] = Det0Proj;
        DetProj2Axis[tx + 1] = Det1Proj;
        __syncthreads();
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
        float DetInterval = VirDetSize * abs(sinVal);
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float Bound0 = max(Point0Proj, DetProj2Axis[0]);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxd = floor((Bound0Proj2Det - Det0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Pointy = (idxrow + 0.5) * ImageSizeY + Point0y;
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                float temp = 0;
                while(idxrow < Height && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        temp += (Point1Proj - Bound0) * ProjTemp[idxd];
                        if (FBPWEIGHT) temp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        temp /= PixInterval;
                        atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, temp);
                        temp = 0;
                        Bound0 = Point1Proj;
                        idxrow++;
                        Pointy += ImageSizeY;
                        Point1Proj += PixInterval;
                    } else {
                        temp += (Det1Proj - Bound0) * ProjTemp[idxd];
                        Bound0 = Det1Proj;
                        idxd ++;
                        if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) {
                    if (FBPWEIGHT) temp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                    temp /= PixInterval;
                    atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, temp);
                }
            }
        }
    }
}


template<bool FBPWEIGHT>
__global__ void BackProjTransFanFlatDisCUDA2dKernel(
    float* __restrict__ Projection,
    cudaTextureObject_t texObj,
    const float* __restrict__ ViewAngle,
    const int Width,
    const int Height,
    const int NumView,
    const int NumDet,
    const float ImageSizeX,
    const float ImageSizeY,
    const float DetSize,
    const float IsoSource,
    const float SourceDetector,
    const float PixXShift,
    const float PixYShift,
    const float BinShift) {

    const int idxBatch = blockIdx.x;
    const int idxView = blockIdx.y;
    const int idxDet0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idxDet = idxDet0 + tx;
    const int MaxNDet = ((idxDet0 + blockDim.x) > NumDet) ? (NumDet - idxDet0) : blockDim.x;
    __shared__ float ProjTemp[BLOCK_DIM];
    __shared__ float DetProj2Axis[BLOCK_DIM + 1];

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float sourcex = - sinVal * IsoSource;
    float sourcey = cosVal * IsoSource;
    float VirDetSize = IsoSource / SourceDetector * DetSize;
    float VirBinShift = IsoSource / SourceDetector * BinShift;
    float Det0x;
    float Det0y;
    float Det1x;
    float Det1y;
    if (cosVal * cosVal > 0.5) {
        ProjTemp[tx] = 0;
        if (cosVal >= 0) {
            Det0x = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
        } else {
            Det0x = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * sinVal;
        }
        float Det0Proj = Map2x(sourcex, sourcey, Det0x, Det0y);
        float Det1Proj = Map2x(sourcex, sourcey, Det1x, Det1y);
        if (tx == 0) DetProj2Axis[tx] = Det0Proj;
        DetProj2Axis[tx + 1] = Det1Proj;
        __syncthreads();
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
        float DetInterval = VirDetSize * abs(cosVal);
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float Bound0 = max(Point0Proj, DetProj2Axis[0]);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxd = floor((Bound0Proj2Det - Det0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Pointx = (idxcol + 0.5) * ImageSizeX + Point0x;
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                float temp = 0;
                while(idxcol < Width && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        float temp2 = (Point1Proj - Bound0) * tex3D<float>(texObj, idxcol, idxrow, idxBatch);
                        if (FBPWEIGHT) temp2 *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        temp += temp2;
                        Bound0 = Point1Proj;
                        idxcol++;
                        Pointx += ImageSizeX;
                        Point1Proj += PixInterval;
                    } else {
                        float temp2 = (Det1Proj - Bound0) * tex3D<float>(texObj, idxcol, idxrow, idxBatch);
                        if (FBPWEIGHT) temp2 *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        temp += temp2;
                        temp /= PixInterval;
                        atomicAdd(ProjTemp + idxd, temp);
                        temp = 0;
                        Bound0 = Det1Proj;
                        idxd ++;
                        if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) {
                    temp /= PixInterval;
                    atomicAdd(ProjTemp + idxd, temp);
                }
            }
        }
         __syncthreads();
        if (idxDet < NumDet) {
            ProjTemp[tx] *= VirDetSize;
            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] = ProjTemp[tx];
        }
    } else {
        ProjTemp[tx] = 0;
        if (sinVal >= 0) {
            Det0x = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((idxDet0 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((idxDet + 1 - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
        } else {
            Det0x = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * cosVal;
            Det0y = ((NumDet / 2.0 - idxDet0) * VirDetSize + VirBinShift) * sinVal;
            Det1x = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * cosVal;
            Det1y = ((NumDet / 2.0 - idxDet - 1) * VirDetSize + VirBinShift) * sinVal;
        }
        float Det0Proj = Map2y(sourcex, sourcey, Det0x, Det0y);
        float Det1Proj = Map2y(sourcex, sourcey, Det1x, Det1y);
        if (tx == 0) DetProj2Axis[tx] = Det0Proj;
        DetProj2Axis[tx + 1] = Det1Proj;
        __syncthreads();
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
        float DetInterval = VirDetSize * abs(sinVal);
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float Bound0 = max(Point0Proj, DetProj2Axis[0]);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxd = floor((Bound0Proj2Det - Det0y) / DetInterval);
                } else {
                    idxd = 0;
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                }
                float Pointy = (idxrow + 0.5) * ImageSizeY + Point0y;
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                float temp = 0;
                while(idxrow < Height && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        float temp2 = (Point1Proj - Bound0) * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch);
                        if (FBPWEIGHT) temp2 *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        temp += temp2;
                        Bound0 = Point1Proj;
                        idxrow++;
                        Pointy += ImageSizeY;
                        Point1Proj += PixInterval;
                    } else {
                        float temp2 = (Det1Proj - Bound0) * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch);
                        if (FBPWEIGHT) temp2 *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        temp += temp2;
                        temp /= PixInterval;
                        atomicAdd(ProjTemp + idxd, temp);
                        temp = 0;
                        Bound0 = Det1Proj;
                        idxd ++;
                        if (idxd < MaxNDet) Det1Proj = DetProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) {
                    temp /= PixInterval;
                    atomicAdd(ProjTemp + idxd, temp);
                }
            }
        }
        __syncthreads();
        if (idxDet < NumDet) {
            ProjTemp[tx] *= VirDetSize;
            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] = ProjTemp[tx];
        }
    }
}


void ProjFanFlatDisCUDA2d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int NumView,
    int NumDet,
    float ImageSizeX,
    float ImageSizeY,
    float DetSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float BinShift) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaExtent extent = make_cudaExtent(Width, Height, BatchSize);
    cudaMalloc3DArray(&cuArray, &channelDesc, extent);

    cudaMemcpy3DParms copyParams={0};
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = make_cudaPitchedPtr((void *)Image, extent.width * sizeof(float), extent.width, extent.height);
    copyParams.dstArray = cuArray;
    cudaMemcpy3D(&copyParams);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    int NumBlockZ = (NumDet - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize, NumView, NumBlockZ);

    ProjFanFlatDisCUDA2dKernel<<<GridSize, BLOCK_DIM>>>(
        Projection, texObj, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}


void ProjTransFanFlatDisCUDA2d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int NumView,
    int NumDet,
    float ImageSizeX,
    float ImageSizeY,
    float DetSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float BinShift) {

    int NumBlockZ = (NumDet - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize, NumView, NumBlockZ);

    ProjTransFanFlatDisCUDA2dKernel<<<GridSize, BLOCK_DIM>>>(
        Image, Projection, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );
}


void BackProjFanFlatDisCUDA2d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int NumView,
    int NumDet,
    float ImageSizeX,
    float ImageSizeY,
    float DetSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float BinShift) {

    int NumBlockZ = (NumDet - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize, NumView, NumBlockZ);

    BackProjFanFlatDisCUDA2dKernel<false><<<GridSize, BLOCK_DIM>>>(
        Image, Projection, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );
}


void BackProjTransFanFlatDisCUDA2d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int NumView,
    int NumDet,
    float ImageSizeX,
    float ImageSizeY,
    float DetSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float BinShift) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaExtent extent = make_cudaExtent(Width, Height, BatchSize);
    cudaMalloc3DArray(&cuArray, &channelDesc, extent);

    cudaMemcpy3DParms copyParams={0};
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = make_cudaPitchedPtr((void *)Image, extent.width * sizeof(float), extent.width, extent.height);
    copyParams.dstArray = cuArray;
    cudaMemcpy3D(&copyParams);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    int NumBlockZ = (NumDet - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize, NumView, NumBlockZ);

    BackProjTransFanFlatDisCUDA2dKernel<false><<<GridSize, BLOCK_DIM>>>(
        Projection, texObj, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}


void BackProjWeightedFanFlatDisCUDA2d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int NumView,
    int NumDet,
    float ImageSizeX,
    float ImageSizeY,
    float DetSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float BinShift) {

    int NumBlockZ = (NumDet - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize, NumView, NumBlockZ);

    BackProjFanFlatDisCUDA2dKernel<true><<<GridSize, BLOCK_DIM>>>(
        Image, Projection, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );
}


void BackProjTransWeightedFanFlatDisCUDA2d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int NumView,
    int NumDet,
    float ImageSizeX,
    float ImageSizeY,
    float DetSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float BinShift) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaExtent extent = make_cudaExtent(Width, Height, BatchSize);
    cudaMalloc3DArray(&cuArray, &channelDesc, extent);

    cudaMemcpy3DParms copyParams={0};
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = make_cudaPitchedPtr((void *)Image, extent.width * sizeof(float), extent.width, extent.height);
    copyParams.dstArray = cuArray;
    cudaMemcpy3D(&copyParams);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    int NumBlockZ = (NumDet - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize, NumView, NumBlockZ);

    BackProjTransFanFlatDisCUDA2dKernel<true><<<GridSize, BLOCK_DIM>>>(
        Projection, texObj, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}
