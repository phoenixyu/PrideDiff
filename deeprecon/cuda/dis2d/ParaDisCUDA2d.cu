#include <cuda.h>

#define BLOCK_DIM 256


template<bool TRANSPOSE>
__global__ void ProjParaDisCUDA2dKernel(
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
    ProjTemp[tx] = 0;
    __syncthreads();

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float Det0Proj;

    if (cosVal * cosVal > 0.5) {
        if (cosVal >= 0) {
            Det0Proj = ((idxDet0 - NumDet / 2.0) * DetSize + BinShift) / cosVal;
        } else {
            Det0Proj = ((NumDet / 2.0 - idxDet0) * DetSize + BinShift) / cosVal;
        }
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float DetInterval = DetSize / abs(cosVal);
        float PixInterval = ImageSizeX;
        float coef = (TRANSPOSE)? (DetSize / PixInterval) : (ImageSizeY / DetSize);
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = sinVal / cosVal * Pointy + Point0x;
                float Bound0 = max(Point0Proj, Det0Proj);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    idxcol = 0;
                    idxd = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxd + 1) * DetInterval + Det0Proj;
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
                        Det1Proj += DetInterval;
                    }
                }
                if (temp != 0) atomicAdd(ProjTemp + idxd, temp);
            }
        }
        __syncthreads();
        if (idxDet < NumDet) {
            ProjTemp[tx] *= coef;
            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] = ProjTemp[tx];
        }
    } else {
        if (sinVal >= 0) {
            Det0Proj = ((idxDet0 - NumDet / 2.0) * DetSize + BinShift) / sinVal;
        } else {
            Det0Proj = ((NumDet / 2.0 - idxDet0) * DetSize + BinShift) / sinVal;
        }
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float DetInterval = DetSize / abs(sinVal);
        float PixInterval = ImageSizeY;
        float coef = (TRANSPOSE)? (DetSize / PixInterval) : (ImageSizeX / DetSize);
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = cosVal / sinVal * Pointx + Point0y;
                float Bound0 = max(Point0Proj, Det0Proj);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    idxrow = 0;
                    idxd = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxd + 1) * DetInterval + Det0Proj;
                float temp = 0;
                while(idxrow < Height && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        temp += (Point1Proj - Bound0) * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch);
                        Bound0 = Point1Proj;
                        idxrow ++;
                        Point1Proj += PixInterval;
                    } else {
                        temp += (Det1Proj - Bound0) * tex3D<float>(texObj, idxcol, Height - 1 - idxrow, idxBatch);
                        atomicAdd(ProjTemp + idxd, temp);
                        temp = 0;
                        Bound0 = Det1Proj;
                        idxd ++;
                        Det1Proj += DetInterval;
                    }
                }
                if (temp != 0) atomicAdd(ProjTemp + idxd, temp);
            }
        }
        __syncthreads();
        if (idxDet < NumDet) {
            ProjTemp[tx] *= coef;
            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] = ProjTemp[tx];
        }
    }
}


template<bool TRANSPOSE>
__global__ void BackProjParaDisCUDA2dKernel(
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

    float sinVal = sin(ViewAngle[idxView]);
    float cosVal = cos(ViewAngle[idxView]);
    float Det0Proj;

    if (cosVal * cosVal > 0.5) {
        if (idxDet < NumDet) {
            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet-1-idxDet);
            ProjTemp[tx] = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
        } else {
            ProjTemp[tx] = 0;
        }
        if (cosVal >= 0) {
            Det0Proj = ((idxDet0 - NumDet / 2.0) * DetSize + BinShift) / cosVal;
        } else {
            Det0Proj = ((NumDet / 2.0 - idxDet0) * DetSize + BinShift) / cosVal;
        }
        float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
        float DetInterval = DetSize / abs(cosVal);
        float PixInterval = ImageSizeX;
        float coef = (TRANSPOSE)? (ImageSizeY / DetSize) : (DetSize / PixInterval);
        ProjTemp[tx] *= coef;
        __syncthreads();
        for (int i = 0; (i * blockDim.x) < Height; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < Height) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = sinVal / cosVal * Pointy + Point0x;
                float Bound0 = max(Point0Proj, Det0Proj);
                int idxcol;
                int idxd;
                if (Point0Proj == Bound0) {
                    idxcol = 0;
                    idxd = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxd + 1) * DetInterval + Det0Proj;
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
                        Det1Proj += DetInterval;
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
            Det0Proj = ((idxDet0 - NumDet / 2.0) * DetSize + BinShift) / sinVal;
        } else {
            Det0Proj = ((NumDet / 2.0 - idxDet0) * DetSize + BinShift) / sinVal;
        }
        float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
        float DetInterval = DetSize / abs(sinVal);
        float PixInterval = ImageSizeY;
        float coef = (TRANSPOSE)? (ImageSizeX / DetSize) : (DetSize / PixInterval);
        ProjTemp[tx] *= coef;
        __syncthreads();
        for (int i = 0; (i * blockDim.x) < Width; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < Width) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = cosVal / sinVal * Pointx + Point0y;
                float Bound0 = max(Point0Proj, Det0Proj);
                int idxrow;
                int idxd;
                if (Point0Proj == Bound0) {
                    idxrow = 0;
                    idxd = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxd = 0;
                }
                float Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxd + 1) * DetInterval + Det0Proj;
                float temp = 0;
                while(idxrow < Height && idxd < MaxNDet) {
                    if (Point1Proj < Det1Proj) {
                        temp += (Point1Proj - Bound0) * ProjTemp[idxd];
                        atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, temp);
                        temp = 0;
                        Bound0 = Point1Proj;
                        idxrow ++;
                        Point1Proj += PixInterval;
                    } else {
                        temp += (Det1Proj - Bound0) * ProjTemp[idxd];
                        Bound0 = Det1Proj;
                        idxd ++;
                        Det1Proj += DetInterval;
                    }
                }
                if (temp != 0) atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, temp);
            }
        }
    }
}


void ProjParaDisCUDA2d(
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

    ProjParaDisCUDA2dKernel<false><<<GridSize, BLOCK_DIM>>>(
        Projection, texObj, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}


void ProjTransParaDisCUDA2d(
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

    BackProjParaDisCUDA2dKernel<true><<<GridSize, BLOCK_DIM>>>(
        Image, Projection, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );
}


void BackProjParaDisCUDA2d(
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

    BackProjParaDisCUDA2dKernel<false><<<GridSize, BLOCK_DIM>>>(
        Image, Projection, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );
}


void BackProjTransParaDisCUDA2d(
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

    ProjParaDisCUDA2dKernel<true><<<GridSize, BLOCK_DIM>>>(
        Projection, texObj, ViewAngle, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
        DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
    );

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}

