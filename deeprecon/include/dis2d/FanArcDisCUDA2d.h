#pragma once

void ProjFanArcDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void ProjTransFanArcDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjFanArcDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjTransFanArcDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjWeightedFanArcDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjTransWeightedFanArcDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);