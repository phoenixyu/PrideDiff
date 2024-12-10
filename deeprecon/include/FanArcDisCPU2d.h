#pragma once

void ProjFanArcDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void ProjTransFanArcDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjFanArcDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjTransFanArcDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjWeightedFanArcDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjTransWeightedFanArcDisCPU2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void GetSysMatrixFanArc2dKernel(std::vector<int64_t>& idxi, std::vector<int64_t>& idxj, std::vector<float>& value, const float* ViewAngle,
    const int Width, const int Height, const int NumView, const int NumDet, const float ImageSizeX, const float ImageSizeY, const float DetSize,
    const float IsoSource, const float SourceDetector, const float PixXShift, const float PixYShift, const float BinShift);