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

__device__ float TriAngCos(float a, float b, float c) {
    return abs(c) / sqrt(a * a + b * b + c * c);
}
}


__global__ void ProjConeFlatDisCUDA3dKernel(
    float* __restrict__ Projection,
    cudaTextureObject_t texObj,
    const float* __restrict__ ViewAngle,
    const int Width,
    const int Height,
    const int Depth,
    const int NumView,
    const int NumDetCol,
    const int NumDetRow,
    const float ImageSizeX,
    const float ImageSizeY,
    const float ImageSizeZ,
    const float DetColSize,
    const float DetRowSize,
    const float IsoSource,
    const float SourceDetector,
    const float PixXShift,
    const float PixYShift,
    const float PixZShift,
    const float BinColShift,
    const float BinRowShift) {

    const int idxBatch = blockIdx.x / NumView;
    const int idxView = blockIdx.x % NumView;
    const int idxDetRow = blockIdx.y;
    const int idxDetCol0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idxDetCol = idxDetCol0 + tx;
    const int MaxNDetCol = ((idxDetCol0 + blockDim.x) > NumDetCol) ? (NumDetCol - idxDetCol0) : blockDim.x;
    __shared__ float ProjTemp[BLOCK_DIM];
    __shared__ float DetRow0Proj2Axis[BLOCK_DIM];
    __shared__ float DetRow1Proj2Axis[BLOCK_DIM];
    __shared__ float DetColProj2Axis[BLOCK_DIM + 1];

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float sourcex = - sinVal * IsoSource;
    float sourcey = cosVal * IsoSource;
    float VirDetColSize = IsoSource / SourceDetector * DetColSize;
    float VirBinColShift = IsoSource / SourceDetector * BinColShift;
    float DetRow0z = ((idxDetRow - NumDetRow / 2.0) * DetRowSize + BinRowShift) * IsoSource / SourceDetector;
    float DetRow1z = DetRow0z + IsoSource / SourceDetector * DetRowSize;
    float DetCol0x;
    float DetCol0y;
    float DetCol1x;
    float DetCol1y;
    if (cosVal * cosVal > 0.5) {
        ProjTemp[tx] = 0;
        float DetColy;
        if (cosVal >= 0) {
            DetCol0x = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetColy = ((idxDetCol + 0.5 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
        } else {
            DetCol0x = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * sinVal;
            DetColy = ((NumDetCol / 2.0 - idxDetCol - 0.5) * VirDetColSize + VirBinColShift) * sinVal;
        }
        float DetCol0Proj = Map2x(sourcex, sourcey, DetCol0x, DetCol0y);
        float DetCol1Proj = Map2x(sourcex, sourcey, DetCol1x, DetCol1y);
        float DetRow0Proj = sourcey / (sourcey - DetColy) * DetRow0z;
        float DetRow1Proj = sourcey / (sourcey - DetColy) * DetRow1z;
        if (tx == 0) DetColProj2Axis[tx] = DetCol0Proj;
        DetColProj2Axis[tx + 1] = DetCol1Proj;
        DetRow0Proj2Axis[tx] = DetRow0Proj;
        DetRow1Proj2Axis[tx] = DetRow1Proj;
        __syncthreads();
        float coef1 = (DetCol1Proj - DetColProj2Axis[tx]) * (DetRow1Proj - DetRow0Proj);
        float coef2 = TriAngCos((DetCol1Proj + DetColProj2Axis[tx]) / 2.0 - sourcex, (DetRow1Proj + DetRow0Proj) / 2.0, sourcey);
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
        float Point0z = - Depth / 2.0 * ImageSizeZ + PixZShift;
        float Point1z = Depth / 2.0 * ImageSizeZ + PixZShift;
        float DetInterval = VirDetColSize * abs(cosVal);
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float Point0Proj2z = sourcey / (sourcey - Pointy) * Point0z;
                float Point1Proj2z = sourcey / (sourcey - Pointy) * Point1z;
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float PixIntervalz = (Point1Proj2z - Point0Proj2z) / Depth;
                float Bound0 = max(Point0Proj, DetColProj2Axis[0]);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxd = floor((Bound0Proj2Det - DetCol0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                float temp = 0;
                while(idxcol < Width && idxd < MaxNDetCol) {
                    float Bound0z = max(Point0Proj2z, DetRow0Proj2Axis[idxd]);
                    float Bound1z = min(Point1Proj2z, DetRow1Proj2Axis[idxd]);
                    int idxslice;
                    if (Bound0z == Point0Proj2z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((Bound0z - Point0Proj2z) / PixIntervalz);
                    }
                    float Point1Proj2z = (idxslice + 1) * PixIntervalz + Point0Proj2z;
                    if (Point1Proj < DetCol1Proj) {
                        float coef = Point1Proj - Bound0;
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            temp += (Point1Proj2z - Bound0z) * coef * tex3D<float>(texObj, idxcol, idxrow, idxBatch * Depth + idxslice);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = Point1Proj;
                        idxcol++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = DetCol1Proj - Bound0;
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            temp += (Point1Proj2z - Bound0z) * coef * tex3D<float>(texObj, idxcol, idxrow, idxBatch * Depth + idxslice);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        atomicAdd(ProjTemp + idxd, temp);
                        temp = 0;
                        Bound0 = DetCol1Proj;
                        idxd ++;
                        if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(ProjTemp + idxd, temp);
            }
        }
        __syncthreads();
        if (idxDetCol < NumDetCol) {
            ProjTemp[tx] *= ImageSizeY / (coef1 * coef2);
            int idxDetColTemp = (cosVal >= 0) ? idxDetCol : (NumDetCol - 1 - idxDetCol);
            Projection[(blockIdx.x * NumDetRow + idxDetRow) * NumDetCol + idxDetColTemp] = ProjTemp[tx];
        }
    } else {
        ProjTemp[tx] = 0;
        float DetColx;
        if (sinVal >= 0) {
            DetCol0x = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetColx = ((idxDetCol + 0.5 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
        } else {
            DetCol0x = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * sinVal;
            DetColx = ((NumDetCol / 2.0 - idxDetCol - 0.5) * VirDetColSize + VirBinColShift) * cosVal;
        }
        float DetCol0Proj = Map2y(sourcex, sourcey, DetCol0x, DetCol0y);
        float DetCol1Proj = Map2y(sourcex, sourcey, DetCol1x, DetCol1y);
        float DetRow0Proj = sourcex / (sourcex - DetColx) * DetRow0z;
        float DetRow1Proj = sourcex / (sourcex - DetColx) * DetRow1z;
        if (tx == 0) DetColProj2Axis[tx] = DetCol0Proj;
        DetColProj2Axis[tx + 1] = DetCol1Proj;
        DetRow0Proj2Axis[tx] = DetRow0Proj;
        DetRow1Proj2Axis[tx] = DetRow1Proj;
        __syncthreads();
        float coef1 = (DetCol1Proj - DetColProj2Axis[tx]) * (DetRow1Proj - DetRow0Proj);
        float coef2 = TriAngCos((DetCol1Proj + DetColProj2Axis[tx]) / 2.0 - sourcey, (DetRow1Proj + DetRow0Proj) / 2.0, sourcex);
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
        float Point0z = - Depth / 2.0 * ImageSizeZ + PixZShift;
        float Point1z = Depth / 2.0 * ImageSizeZ + PixZShift;
        float DetInterval = VirDetColSize * abs(sinVal);
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float Point0Proj2z = sourcex / (sourcex - Pointx) * Point0z;
                float Point1Proj2z = sourcex / (sourcex - Pointx) * Point1z;
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float PixIntervalz = (Point1Proj2z - Point0Proj2z) / Depth;
                float Bound0 = max(Point0Proj, DetColProj2Axis[0]);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxd = floor((Bound0Proj2Det - DetCol0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                float temp = 0;
                while(idxrow < Height && idxd < MaxNDetCol) {
                    float Bound0z = max(Point0Proj2z, DetRow0Proj2Axis[idxd]);
                    float Bound1z = min(Point1Proj2z, DetRow1Proj2Axis[idxd]);
                    int idxslice;
                    if (Bound0z == Point0Proj2z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((Bound0z - Point0Proj2z) / PixIntervalz);
                    }
                    float Point1Proj2z = (idxslice + 1) * PixIntervalz + Point0Proj2z;
                    if (Point1Proj < DetCol1Proj) {
                        float coef = Point1Proj - Bound0;
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            temp += (Point1Proj2z - Bound0z) * coef * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch * Depth + idxslice);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = Point1Proj;
                        idxrow++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = DetCol1Proj - Bound0;
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            temp += (Point1Proj2z - Bound0z) * coef * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch * Depth + idxslice);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        atomicAdd(ProjTemp + idxd, temp);
                        temp = 0;
                        Bound0 = DetCol1Proj;
                        idxd ++;
                        if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(ProjTemp + idxd, temp);
            }
        }
        __syncthreads();
        if (idxDetCol < NumDetCol) {
            ProjTemp[tx] *= ImageSizeX / (coef1 * coef2);
            int idxDetColTemp = (sinVal >= 0) ? idxDetCol : (NumDetCol - 1 - idxDetCol);
            Projection[(blockIdx.x * NumDetRow + idxDetRow) * NumDetCol + idxDetColTemp] = ProjTemp[tx];
        }
    }
}


__global__ void ProjTransConeFlatDisCUDA3dKernel(
    float* __restrict__ Image,
    const float* __restrict__ Projection,
    const float* __restrict__ ViewAngle,
    const int Width,
    const int Height,
    const int Depth,
    const int NumView,
    const int NumDetCol,
    const int NumDetRow,
    const float ImageSizeX,
    const float ImageSizeY,
    const float ImageSizeZ,
    const float DetColSize,
    const float DetRowSize,
    const float IsoSource,
    const float SourceDetector,
    const float PixXShift,
    const float PixYShift,
    const float PixZShift,
    const float BinColShift,
    const float BinRowShift) {

    const int idxBatch = blockIdx.x / NumView;
    const int idxView = blockIdx.x % NumView;
    const int idxDetRow = blockIdx.y;
    const int idxDetCol0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idxDetCol = idxDetCol0 + tx;
    const int MaxNDetCol = ((idxDetCol0 + blockDim.x) > NumDetCol) ? (NumDetCol - idxDetCol0) : blockDim.x;
    __shared__ float ProjTemp[BLOCK_DIM];
    __shared__ float DetRow0Proj2Axis[BLOCK_DIM];
    __shared__ float DetRow1Proj2Axis[BLOCK_DIM];
    __shared__ float DetColProj2Axis[BLOCK_DIM + 1];

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float sourcex = - sinVal * IsoSource;
    float sourcey = cosVal * IsoSource;
    float VirDetColSize = IsoSource / SourceDetector * DetColSize;
    float VirBinColShift = IsoSource / SourceDetector * BinColShift;
    float DetRow0z = ((idxDetRow - NumDetRow / 2.0) * DetRowSize + BinRowShift) * IsoSource / SourceDetector;
    float DetRow1z = DetRow0z + IsoSource / SourceDetector * DetRowSize;
    float DetCol0x;
    float DetCol0y;
    float DetCol1x;
    float DetCol1y;
    if (cosVal * cosVal > 0.5) {
        if (idxDetCol < NumDetCol) {
            int idxDetColTemp = (cosVal >= 0) ? idxDetCol : (NumDetCol - 1 - idxDetCol);
            ProjTemp[tx] = Projection[(blockIdx.x * NumDetRow + idxDetRow) * NumDetCol + idxDetColTemp];
        } else {
            ProjTemp[tx] = 0;
        }
        float DetColy;
        if (cosVal >= 0) {
            DetCol0x = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetColy = ((idxDetCol + 0.5 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
        } else {
            DetCol0x = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * sinVal;
            DetColy = ((NumDetCol / 2.0 - idxDetCol - 0.5) * VirDetColSize + VirBinColShift) * sinVal;
        }
        float DetCol0Proj = Map2x(sourcex, sourcey, DetCol0x, DetCol0y);
        float DetCol1Proj = Map2x(sourcex, sourcey, DetCol1x, DetCol1y);
        float DetRow0Proj = sourcey / (sourcey - DetColy) * DetRow0z;
        float DetRow1Proj = sourcey / (sourcey - DetColy) * DetRow1z;
        if (tx == 0) DetColProj2Axis[tx] = DetCol0Proj;
        DetColProj2Axis[tx + 1] = DetCol1Proj;
        DetRow0Proj2Axis[tx] = DetRow0Proj;
        DetRow1Proj2Axis[tx] = DetRow1Proj;
        __syncthreads();
        float coef1 = (DetCol1Proj - DetColProj2Axis[tx]) * (DetRow1Proj - DetRow0Proj);
        float coef2 = TriAngCos((DetCol1Proj + DetColProj2Axis[tx]) / 2.0 - sourcex, (DetRow1Proj + DetRow0Proj) / 2.0, sourcey);
        ProjTemp[tx] *= ImageSizeY / (coef1 * coef2);
        __syncthreads();
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
        float Point0z = - Depth / 2.0 * ImageSizeZ + PixZShift;
        float Point1z = Depth / 2.0 * ImageSizeZ + PixZShift;
        float DetInterval = VirDetColSize * abs(cosVal);
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float Point0Proj2z = sourcey / (sourcey - Pointy) * Point0z;
                float Point1Proj2z = sourcey / (sourcey - Pointy) * Point1z;
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float PixIntervalz = (Point1Proj2z - Point0Proj2z) / Depth;
                float Bound0 = max(Point0Proj, DetColProj2Axis[0]);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxd = floor((Bound0Proj2Det - DetCol0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                while(idxcol < Width && idxd < MaxNDetCol) {
                    float Bound0z = max(Point0Proj2z, DetRow0Proj2Axis[idxd]);
                    float Bound1z = min(Point1Proj2z, DetRow1Proj2Axis[idxd]);
                    int idxslice;
                    if (Bound0z == Point0Proj2z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((Bound0z - Point0Proj2z) / PixIntervalz);
                    }
                    float Point1Proj2z = (idxslice + 1) * PixIntervalz + Point0Proj2z;
                    if (Point1Proj < DetCol1Proj) {
                        float coef = Point1Proj - Bound0;
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            float temp = (Point1Proj2z - Bound0z) * coef * ProjTemp[idxd];
                            atomicAdd(Image + ((idxBatch * Depth + idxslice) * Height + idxrow) * Width + idxcol, temp);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = Point1Proj;
                        idxcol++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = DetCol1Proj - Bound0;
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            float temp = (Point1Proj2z - Bound0z) * coef * ProjTemp[idxd];
                            atomicAdd(Image + ((idxBatch * Depth + idxslice) * Height + idxrow) * Width + idxcol, temp);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = DetCol1Proj;
                        idxd ++;
                        if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                    }
                }
            }
        }
    } else {
        if (idxDetCol < NumDetCol) {
            int idxDetColTemp = (sinVal >= 0) ? idxDetCol : (NumDetCol - 1 - idxDetCol);
            ProjTemp[tx] = Projection[(blockIdx.x * NumDetRow + idxDetRow) * NumDetCol + idxDetColTemp];
        } else {
            ProjTemp[tx] = 0;
        }
        float DetColx;
        if (sinVal >= 0) {
            DetCol0x = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetColx = ((idxDetCol + 0.5 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
        } else {
            DetCol0x = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * sinVal;
            DetColx = ((NumDetCol / 2.0 - idxDetCol - 0.5) * VirDetColSize + VirBinColShift) * cosVal;
        }
        float DetCol0Proj = Map2y(sourcex, sourcey, DetCol0x, DetCol0y);
        float DetCol1Proj = Map2y(sourcex, sourcey, DetCol1x, DetCol1y);
        float DetRow0Proj = sourcex / (sourcex - DetColx) * DetRow0z;
        float DetRow1Proj = sourcex / (sourcex - DetColx) * DetRow1z;
        if (tx == 0) DetColProj2Axis[tx] = DetCol0Proj;
        DetColProj2Axis[tx + 1] = DetCol1Proj;
        DetRow0Proj2Axis[tx] = DetRow0Proj;
        DetRow1Proj2Axis[tx] = DetRow1Proj;
        __syncthreads();
        float coef1 = (DetCol1Proj - DetColProj2Axis[tx]) * (DetRow1Proj - DetRow0Proj);
        float coef2 = TriAngCos((DetCol1Proj + DetColProj2Axis[tx]) / 2.0 - sourcey, (DetRow1Proj + DetRow0Proj) / 2.0, sourcex);
        ProjTemp[tx] *= ImageSizeX / (coef1 * coef2);
        __syncthreads();
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
        float Point0z = - Depth / 2.0 * ImageSizeZ + PixZShift;
        float Point1z = Depth / 2.0 * ImageSizeZ + PixZShift;
        float DetInterval = VirDetColSize * abs(sinVal);
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float Point0Proj2z = sourcex / (sourcex - Pointx) * Point0z;
                float Point1Proj2z = sourcex / (sourcex - Pointx) * Point1z;
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float PixIntervalz = (Point1Proj2z - Point0Proj2z) / Depth;
                float Bound0 = max(Point0Proj, DetColProj2Axis[0]);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxd = floor((Bound0Proj2Det - DetCol0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                while(idxrow < Height && idxd < MaxNDetCol) {
                    float Bound0z = max(Point0Proj2z, DetRow0Proj2Axis[idxd]);
                    float Bound1z = min(Point1Proj2z, DetRow1Proj2Axis[idxd]);
                    int idxslice;
                    if (Bound0z == Point0Proj2z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((Bound0z - Point0Proj2z) / PixIntervalz);
                    }
                    float Point1Proj2z = (idxslice + 1) * PixIntervalz + Point0Proj2z;
                    if (Point1Proj < DetCol1Proj) {
                        float coef = Point1Proj - Bound0;
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            float temp = (Point1Proj2z - Bound0z) * coef * ProjTemp[idxd];
                            atomicAdd(Image + ((idxBatch * Depth + idxslice) * Height + Height - 1 - idxrow) * Width + idxcol, temp);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = Point1Proj;
                        idxrow++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = DetCol1Proj - Bound0;
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            float temp = (Point1Proj2z - Bound0z) * coef * ProjTemp[idxd];
                            atomicAdd(Image + ((idxBatch * Depth + idxslice) * Height + Height - 1 - idxrow) * Width + idxcol, temp);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = DetCol1Proj;
                        idxd ++;
                        if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                    }
                }
            }
        }
    }
}


template<bool FBPWEIGHT>
__global__ void BackProjConeFlatDisCUDA3dKernel(
    float* __restrict__ Image,
    const float* __restrict__ Projection,
    const float* __restrict__ ViewAngle,
    const int Width,
    const int Height,
    const int Depth,
    const int NumView,
    const int NumDetCol,
    const int NumDetRow,
    const float ImageSizeX,
    const float ImageSizeY,
    const float ImageSizeZ,
    const float DetColSize,
    const float DetRowSize,
    const float IsoSource,
    const float SourceDetector,
    const float PixXShift,
    const float PixYShift,
    const float PixZShift,
    const float BinColShift,
    const float BinRowShift) {

    const int idxBatch = blockIdx.x / NumView;
    const int idxView = blockIdx.x % NumView;
    const int idxDetRow = blockIdx.y;
    const int idxDetCol0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idxDetCol = idxDetCol0 + tx;
    const int MaxNDetCol = ((idxDetCol0 + blockDim.x) > NumDetCol) ? (NumDetCol - idxDetCol0) : blockDim.x;
    __shared__ float ProjTemp[BLOCK_DIM];
    __shared__ float DetRow0Proj2Axis[BLOCK_DIM];
    __shared__ float DetRow1Proj2Axis[BLOCK_DIM];
    __shared__ float DetColProj2Axis[BLOCK_DIM + 1];

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float sourcex = - sinVal * IsoSource;
    float sourcey = cosVal * IsoSource;
    float VirDetColSize = IsoSource / SourceDetector * DetColSize;
    float VirBinColShift = IsoSource / SourceDetector * BinColShift;
    float DetRow0z = ((idxDetRow - NumDetRow / 2.0) * DetRowSize + BinRowShift) * IsoSource / SourceDetector;
    float DetRow1z = DetRow0z + IsoSource / SourceDetector * DetRowSize;
    float DetCol0x;
    float DetCol0y;
    float DetCol1x;
    float DetCol1y;
    if (cosVal * cosVal > 0.5) {
        if (idxDetCol < NumDetCol) {
            int idxDetColTemp = (cosVal >= 0) ? idxDetCol : (NumDetCol - 1 - idxDetCol);
            ProjTemp[tx] = Projection[(blockIdx.x * NumDetRow + idxDetRow) * NumDetCol + idxDetColTemp] * VirDetColSize;
        } else {
            ProjTemp[tx] = 0;
        }
        float DetColy;
        if (cosVal >= 0) {
            DetCol0x = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetColy = ((idxDetCol + 0.5 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
        } else {
            DetCol0x = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * sinVal;
            DetColy = ((NumDetCol / 2.0 - idxDetCol - 0.5) * VirDetColSize + VirBinColShift) * sinVal;
        }
        float DetCol0Proj = Map2x(sourcex, sourcey, DetCol0x, DetCol0y);
        float DetCol1Proj = Map2x(sourcex, sourcey, DetCol1x, DetCol1y);
        float DetRow0Proj = sourcey / (sourcey - DetColy) * DetRow0z;
        float DetRow1Proj = sourcey / (sourcey - DetColy) * DetRow1z;
        if (tx == 0) DetColProj2Axis[tx] = DetCol0Proj;
        DetColProj2Axis[tx + 1] = DetCol1Proj;
        DetRow0Proj2Axis[tx] = DetRow0Proj;
        DetRow1Proj2Axis[tx] = DetRow1Proj;
        __syncthreads();
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
        float Point0z = - Depth / 2.0 * ImageSizeZ + PixZShift;
        float Point1z = Depth / 2.0 * ImageSizeZ + PixZShift;
        float DetInterval = VirDetColSize * abs(cosVal);
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float Point0Proj2z = sourcey / (sourcey - Pointy) * Point0z;
                float Point1Proj2z = sourcey / (sourcey - Pointy) * Point1z;
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float PixIntervalz = (Point1Proj2z - Point0Proj2z) / Depth;
                float PixArea = PixInterval * PixIntervalz;
                float Bound0 = max(Point0Proj, DetColProj2Axis[0]);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxd = floor((Bound0Proj2Det - DetCol0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Pointx = (idxcol + 0.5) * ImageSizeX + Point0x;
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                while(idxcol < Width && idxd < MaxNDetCol) {
                    float Bound0z = max(Point0Proj2z, DetRow0Proj2Axis[idxd]);
                    float Bound1z = min(Point1Proj2z, DetRow1Proj2Axis[idxd]);
                    int idxslice;
                    if (Bound0z == Point0Proj2z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((Bound0z - Point0Proj2z) / PixIntervalz);
                    }
                    float Point1Proj2z = (idxslice + 1) * PixIntervalz + Point0Proj2z;
                    if (Point1Proj < DetCol1Proj) {
                        float coef = (Point1Proj - Bound0) / PixArea;
                        if (FBPWEIGHT) coef *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            float temp = (Point1Proj2z - Bound0z) * coef * ProjTemp[idxd];
                            atomicAdd(Image + ((idxBatch * Depth + idxslice) * Height + idxrow) * Width + idxcol, temp);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = Point1Proj;
                        idxcol++;
                        Pointx += ImageSizeX;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (DetCol1Proj - Bound0) / PixArea;
                        if (FBPWEIGHT) coef *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            float temp = (Point1Proj2z - Bound0z) * coef * ProjTemp[idxd];
                            atomicAdd(Image + ((idxBatch * Depth + idxslice) * Height + idxrow) * Width + idxcol, temp);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = DetCol1Proj;
                        idxd ++;
                        if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                    }
                }
            }
        }
    } else {
        if (idxDetCol < NumDetCol) {
            int idxDetColTemp = (sinVal >= 0) ? idxDetCol : (NumDetCol - 1 - idxDetCol);
            ProjTemp[tx] = Projection[(blockIdx.x * NumDetRow + idxDetRow) * NumDetCol + idxDetColTemp] * VirDetColSize;
        } else {
            ProjTemp[tx] = 0;
        }
        float DetColx;
        if (sinVal >= 0) {
            DetCol0x = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetColx = ((idxDetCol + 0.5 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
        } else {
            DetCol0x = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * sinVal;
            DetColx = ((NumDetCol / 2.0 - idxDetCol - 0.5) * VirDetColSize + VirBinColShift) * cosVal;
        }
        float DetCol0Proj = Map2y(sourcex, sourcey, DetCol0x, DetCol0y);
        float DetCol1Proj = Map2y(sourcex, sourcey, DetCol1x, DetCol1y);
        float DetRow0Proj = sourcex / (sourcex - DetColx) * DetRow0z;
        float DetRow1Proj = sourcex / (sourcex - DetColx) * DetRow1z;
        if (tx == 0) DetColProj2Axis[tx] = DetCol0Proj;
        DetColProj2Axis[tx + 1] = DetCol1Proj;
        DetRow0Proj2Axis[tx] = DetRow0Proj;
        DetRow1Proj2Axis[tx] = DetRow1Proj;
        __syncthreads();
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
        float Point0z = - Depth / 2.0 * ImageSizeZ + PixZShift;
        float Point1z = Depth / 2.0 * ImageSizeZ + PixZShift;
        float DetInterval = VirDetColSize * abs(sinVal);
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float Point0Proj2z = sourcex / (sourcex - Pointx) * Point0z;
                float Point1Proj2z = sourcex / (sourcex - Pointx) * Point1z;
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float PixIntervalz = (Point1Proj2z - Point0Proj2z) / Depth;
                float PixArea = PixInterval * PixIntervalz;
                float Bound0 = max(Point0Proj, DetColProj2Axis[0]);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxd = floor((Bound0Proj2Det - DetCol0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Pointy = (idxrow + 0.5) * ImageSizeY + Point0y;
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                while(idxrow < Height && idxd < MaxNDetCol) {
                    float Bound0z = max(Point0Proj2z, DetRow0Proj2Axis[idxd]);
                    float Bound1z = min(Point1Proj2z, DetRow1Proj2Axis[idxd]);
                    int idxslice;
                    if (Bound0z == Point0Proj2z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((Bound0z - Point0Proj2z) / PixIntervalz);
                    }
                    float Point1Proj2z = (idxslice + 1) * PixIntervalz + Point0Proj2z;
                    if (Point1Proj < DetCol1Proj) {
                        float coef = (Point1Proj - Bound0) / PixArea;
                        if (FBPWEIGHT) coef *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            float temp = (Point1Proj2z - Bound0z) * coef * ProjTemp[idxd];
                            atomicAdd(Image + ((idxBatch * Depth + idxslice) * Height + Height - 1 - idxrow) * Width + idxcol, temp);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = Point1Proj;
                        idxrow++;
                        Pointy += ImageSizeY;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (DetCol1Proj - Bound0) / PixArea;
                        if (FBPWEIGHT) coef *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            float temp = (Point1Proj2z - Bound0z) * coef * ProjTemp[idxd];
                            atomicAdd(Image + ((idxBatch * Depth + idxslice) * Height + Height - 1 - idxrow) * Width + idxcol, temp);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = DetCol1Proj;
                        idxd ++;
                        if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                    }
                }
            }
        }
    }
}


template<bool FBPWEIGHT>
__global__ void BackProjTransConeFlatDisCUDA3dKernel(
    float* __restrict__ Projection,
    cudaTextureObject_t texObj,
    const float* __restrict__ ViewAngle,
    const int Width,
    const int Height,
    const int Depth,
    const int NumView,
    const int NumDetCol,
    const int NumDetRow,
    const float ImageSizeX,
    const float ImageSizeY,
    const float ImageSizeZ,
    const float DetColSize,
    const float DetRowSize,
    const float IsoSource,
    const float SourceDetector,
    const float PixXShift,
    const float PixYShift,
    const float PixZShift,
    const float BinColShift,
    const float BinRowShift) {

    const int idxBatch = blockIdx.x / NumView;
    const int idxView = blockIdx.x % NumView;
    const int idxDetRow = blockIdx.y;
    const int idxDetCol0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idxDetCol = idxDetCol0 + tx;
    const int MaxNDetCol = ((idxDetCol0 + blockDim.x) > NumDetCol) ? (NumDetCol - idxDetCol0) : blockDim.x;
    __shared__ float ProjTemp[BLOCK_DIM];
    __shared__ float DetRow0Proj2Axis[BLOCK_DIM];
    __shared__ float DetRow1Proj2Axis[BLOCK_DIM];
    __shared__ float DetColProj2Axis[BLOCK_DIM + 1];

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float sourcex = - sinVal * IsoSource;
    float sourcey = cosVal * IsoSource;
    float VirDetColSize = IsoSource / SourceDetector * DetColSize;
    float VirBinColShift = IsoSource / SourceDetector * BinColShift;
    float DetRow0z = ((idxDetRow - NumDetRow / 2.0) * DetRowSize + BinRowShift) * IsoSource / SourceDetector;
    float DetRow1z = DetRow0z + IsoSource / SourceDetector * DetRowSize;
    float DetCol0x;
    float DetCol0y;
    float DetCol1x;
    float DetCol1y;
    if (cosVal * cosVal > 0.5) {
        ProjTemp[tx] = 0;
        float DetColy;
        if (cosVal >= 0) {
            DetCol0x = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetColy = ((idxDetCol + 0.5 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
        } else {
            DetCol0x = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * sinVal;
            DetColy = ((NumDetCol / 2.0 - idxDetCol - 0.5) * VirDetColSize + VirBinColShift) * sinVal;
        }
        float DetCol0Proj = Map2x(sourcex, sourcey, DetCol0x, DetCol0y);
        float DetCol1Proj = Map2x(sourcex, sourcey, DetCol1x, DetCol1y);
        float DetRow0Proj = sourcey / (sourcey - DetColy) * DetRow0z;
        float DetRow1Proj = sourcey / (sourcey - DetColy) * DetRow1z;
        if (tx == 0) DetColProj2Axis[tx] = DetCol0Proj;
        DetColProj2Axis[tx + 1] = DetCol1Proj;
        DetRow0Proj2Axis[tx] = DetRow0Proj;
        DetRow1Proj2Axis[tx] = DetRow1Proj;
        __syncthreads();
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
        float Point0z = - Depth / 2.0 * ImageSizeZ + PixZShift;
        float Point1z = Depth / 2.0 * ImageSizeZ + PixZShift;
        float DetInterval = VirDetColSize * abs(cosVal);
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float Point0Proj2z = sourcey / (sourcey - Pointy) * Point0z;
                float Point1Proj2z = sourcey / (sourcey - Pointy) * Point1z;
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float PixIntervalz = (Point1Proj2z - Point0Proj2z) / Depth;
                float PixArea = PixInterval * PixIntervalz;
                float Bound0 = max(Point0Proj, DetColProj2Axis[0]);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxd = floor((Bound0Proj2Det - DetCol0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Pointx = (idxcol + 0.5) * ImageSizeX + Point0x;
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                float temp = 0;
                while(idxcol < Width && idxd < MaxNDetCol) {
                    float Bound0z = max(Point0Proj2z, DetRow0Proj2Axis[idxd]);
                    float Bound1z = min(Point1Proj2z, DetRow1Proj2Axis[idxd]);
                    int idxslice;
                    if (Bound0z == Point0Proj2z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((Bound0z - Point0Proj2z) / PixIntervalz);
                    }
                    float Point1Proj2z = (idxslice + 1) * PixIntervalz + Point0Proj2z;
                    if (Point1Proj < DetCol1Proj) {
                        float coef = (Point1Proj - Bound0) / PixArea;
                        if (FBPWEIGHT) coef *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            temp += (Point1Proj2z - Bound0z) * coef * tex3D<float>(texObj, idxcol, idxrow, idxBatch * Depth + idxslice);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = Point1Proj;
                        idxcol++;
                        Pointx += ImageSizeX;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (DetCol1Proj - Bound0) / PixArea;
                        if (FBPWEIGHT) coef *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            temp += (Point1Proj2z - Bound0z) * coef * tex3D<float>(texObj, idxcol, idxrow, idxBatch * Depth + idxslice);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        atomicAdd(ProjTemp + idxd, temp);
                        temp = 0;
                        Bound0 = DetCol1Proj;
                        idxd ++;
                        if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(ProjTemp + idxd, temp);
            }
        }
        __syncthreads();
        if (idxDetCol < NumDetCol) {
            ProjTemp[tx] *= VirDetColSize;
            int idxDetColTemp = (cosVal >= 0) ? idxDetCol : (NumDetCol - 1 - idxDetCol);
            Projection[(blockIdx.x * NumDetRow + idxDetRow) * NumDetCol + idxDetColTemp] = ProjTemp[tx];
        }
    } else {
        ProjTemp[tx] = 0;
        float DetColx;
        if (sinVal >= 0) {
            DetCol0x = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((idxDetCol0 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((idxDetCol + 1 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * sinVal;
            DetColx = ((idxDetCol + 0.5 - NumDetCol / 2.0) * VirDetColSize + VirBinColShift) * cosVal;
        } else {
            DetCol0x = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol0y = ((NumDetCol / 2.0 - idxDetCol0) * VirDetColSize + VirBinColShift) * sinVal;
            DetCol1x = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * cosVal;
            DetCol1y = ((NumDetCol / 2.0 - idxDetCol - 1) * VirDetColSize + VirBinColShift) * sinVal;
            DetColx = ((NumDetCol / 2.0 - idxDetCol - 0.5) * VirDetColSize + VirBinColShift) * cosVal;
        }
        float DetCol0Proj = Map2y(sourcex, sourcey, DetCol0x, DetCol0y);
        float DetCol1Proj = Map2y(sourcex, sourcey, DetCol1x, DetCol1y);
        float DetRow0Proj = sourcex / (sourcex - DetColx) * DetRow0z;
        float DetRow1Proj = sourcex / (sourcex - DetColx) * DetRow1z;
        if (tx == 0) DetColProj2Axis[tx] = DetCol0Proj;
        DetColProj2Axis[tx + 1] = DetCol1Proj;
        DetRow0Proj2Axis[tx] = DetRow0Proj;
        DetRow1Proj2Axis[tx] = DetRow1Proj;
        __syncthreads();
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
        float Point0z = - Depth / 2.0 * ImageSizeZ + PixZShift;
        float Point1z = Depth / 2.0 * ImageSizeZ + PixZShift;
        float DetInterval = VirDetColSize * abs(sinVal);
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float Point0Proj2z = sourcex / (sourcex - Pointx) * Point0z;
                float Point1Proj2z = sourcex / (sourcex - Pointx) * Point1z;
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float PixIntervalz = (Point1Proj2z - Point0Proj2z) / Depth;
                float PixArea = PixInterval * PixIntervalz;
                float Bound0 = max(Point0Proj, DetColProj2Axis[0]);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxd = floor((Bound0Proj2Det - DetCol0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Pointy = (idxrow + 0.5) * ImageSizeY + Point0y;
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                float temp = 0;
                while(idxrow < Height && idxd < MaxNDetCol) {
                    float Bound0z = max(Point0Proj2z, DetRow0Proj2Axis[idxd]);
                    float Bound1z = min(Point1Proj2z, DetRow1Proj2Axis[idxd]);
                    int idxslice;
                    if (Bound0z == Point0Proj2z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((Bound0z - Point0Proj2z) / PixIntervalz);
                    }
                    float Point1Proj2z = (idxslice + 1) * PixIntervalz + Point0Proj2z;
                    if (Point1Proj < DetCol1Proj) {
                        float coef = (Point1Proj - Bound0) / PixArea;
                        if (FBPWEIGHT) coef *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            temp += (Point1Proj2z - Bound0z) * coef * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch * Depth + idxslice);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        Bound0 = Point1Proj;
                        idxrow++;
                        Pointy += ImageSizeY;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (DetCol1Proj - Bound0) / PixArea;
                        if (FBPWEIGHT) coef *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                        while (Bound0z < Bound1z && idxslice < Depth) {
                            Point1Proj2z = (Point1Proj2z > Bound1z) ? Bound1z : Point1Proj2z;
                            temp += (Point1Proj2z - Bound0z) * coef * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch * Depth + idxslice);
                            Bound0z = Point1Proj2z;
                            idxslice++;
                            Point1Proj2z += PixIntervalz;
                        }
                        atomicAdd(ProjTemp + idxd, temp);
                        temp = 0;
                        Bound0 = DetCol1Proj;
                        idxd ++;
                        if (idxd < MaxNDetCol) DetCol1Proj = DetColProj2Axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(ProjTemp + idxd, temp);
            }
        }
        __syncthreads();
        if (idxDetCol < NumDetCol) {
            ProjTemp[tx] *= VirDetColSize;
            int idxDetColTemp = (sinVal >= 0) ? idxDetCol : (NumDetCol - 1 - idxDetCol);
            Projection[(blockIdx.x * NumDetRow + idxDetRow) * NumDetCol + idxDetColTemp] = ProjTemp[tx];
        }
    }
}


void ProjConeFlatDisCUDA3d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int Depth,
    int NumView,
    int NumDetCol,
    int NumDetRow,
    float ImageSizeX,
    float ImageSizeY,
    float ImageSizeZ,
    float DetColSize,
    float DetRowSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float PixZShift,
    float BinColShift,
    float BinRowShift) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaExtent extent = make_cudaExtent(Width, Height, BatchSize * Depth);
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

    int NumBlockZ = (NumDetCol - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize * NumView, NumDetRow, NumBlockZ);

    ProjConeFlatDisCUDA3dKernel<<<GridSize, BLOCK_DIM>>>(
        Projection, texObj, ViewAngle, Width, Height, Depth, NumView, NumDetCol, NumDetRow, ImageSizeX, ImageSizeY, ImageSizeZ,
        DetColSize, DetRowSize, IsoSource, SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
    );

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}


void ProjTransConeFlatDisCUDA3d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int Depth,
    int NumView,
    int NumDetCol,
    int NumDetRow,
    float ImageSizeX,
    float ImageSizeY,
    float ImageSizeZ,
    float DetColSize,
    float DetRowSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float PixZShift,
    float BinColShift,
    float BinRowShift) {

    int NumBlockZ = (NumDetCol - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize * NumView, NumDetRow, NumBlockZ);

    ProjTransConeFlatDisCUDA3dKernel<<<GridSize, BLOCK_DIM>>>(
        Image, Projection, ViewAngle, Width, Height, Depth, NumView, NumDetCol, NumDetRow, ImageSizeX, ImageSizeY, ImageSizeZ,
        DetColSize, DetRowSize, IsoSource, SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
    );
}


void BackProjConeFlatDisCUDA3d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int Depth,
    int NumView,
    int NumDetCol,
    int NumDetRow,
    float ImageSizeX,
    float ImageSizeY,
    float ImageSizeZ,
    float DetColSize,
    float DetRowSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float PixZShift,
    float BinColShift,
    float BinRowShift) {

    int NumBlockZ = (NumDetCol - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize * NumView, NumDetRow, NumBlockZ);

    BackProjConeFlatDisCUDA3dKernel<false><<<GridSize, BLOCK_DIM>>>(
        Image, Projection, ViewAngle, Width, Height, Depth, NumView, NumDetCol, NumDetRow, ImageSizeX, ImageSizeY, ImageSizeZ,
        DetColSize, DetRowSize, IsoSource, SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
    );
}


void BackProjTransConeFlatDisCUDA3d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int Depth,
    int NumView,
    int NumDetCol,
    int NumDetRow,
    float ImageSizeX,
    float ImageSizeY,
    float ImageSizeZ,
    float DetColSize,
    float DetRowSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float PixZShift,
    float BinColShift,
    float BinRowShift) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaExtent extent = make_cudaExtent(Width, Height, BatchSize * Depth);
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

    int NumBlockZ = (NumDetCol - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize * NumView, NumDetRow, NumBlockZ);

    BackProjTransConeFlatDisCUDA3dKernel<false><<<GridSize, BLOCK_DIM>>>(
        Projection, texObj, ViewAngle, Width, Height, Depth, NumView, NumDetCol, NumDetRow, ImageSizeX, ImageSizeY, ImageSizeZ,
        DetColSize, DetRowSize, IsoSource, SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
    );

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}


void BackProjWeightedConeFlatDisCUDA3d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int Depth,
    int NumView,
    int NumDetCol,
    int NumDetRow,
    float ImageSizeX,
    float ImageSizeY,
    float ImageSizeZ,
    float DetColSize,
    float DetRowSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float PixZShift,
    float BinColShift,
    float BinRowShift) {

    int NumBlockZ = (NumDetCol - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize * NumView, NumDetRow, NumBlockZ);

    BackProjConeFlatDisCUDA3dKernel<true><<<GridSize, BLOCK_DIM>>>(
        Image, Projection, ViewAngle, Width, Height, Depth, NumView, NumDetCol, NumDetRow, ImageSizeX, ImageSizeY, ImageSizeZ,
        DetColSize, DetRowSize, IsoSource, SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
    );
}


void BackProjTransWeightedConeFlatDisCUDA3d(
    float *Image,
    float *Projection,
    float *ViewAngle,
    int BatchSize,
    int Width,
    int Height,
    int Depth,
    int NumView,
    int NumDetCol,
    int NumDetRow,
    float ImageSizeX,
    float ImageSizeY,
    float ImageSizeZ,
    float DetColSize,
    float DetRowSize,
    float IsoSource,
    float SourceDetector,
    float PixXShift,
    float PixYShift,
    float PixZShift,
    float BinColShift,
    float BinRowShift) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaExtent extent = make_cudaExtent(Width, Height, BatchSize * Depth);
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

    int NumBlockZ = (NumDetCol - 1) / BLOCK_DIM + 1;
    const dim3 GridSize(BatchSize * NumView, NumDetRow, NumBlockZ);

    BackProjTransConeFlatDisCUDA3dKernel<true><<<GridSize, BLOCK_DIM>>>(
        Projection, texObj, ViewAngle, Width, Height, Depth, NumView, NumDetCol, NumDetRow, ImageSizeX, ImageSizeY, ImageSizeZ,
        DetColSize, DetRowSize, IsoSource, SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
    );

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}