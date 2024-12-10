#pragma once


void ProjParaDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void ProjTransParaDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjParaDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjTransParaDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void GetSysMatrixPara2dKernel(std::vector<int64_t>& idxi, std::vector<int64_t>& idxj, std::vector<float>& value, const float* ViewAngle,
    const int Width, const int Height, const int NumView, const int NumDet, const float ImageSizeX, const float ImageSizeY, const float DetSize,
    const float IsoSource, const float SourceDetector, const float PixXShift, const float PixYShift, const float BinShift);