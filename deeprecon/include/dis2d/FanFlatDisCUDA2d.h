#pragma once

void ProjFanFlatDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void ProjTransFanFlatDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjFanFlatDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjTransFanFlatDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjWeightedFanFlatDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjTransWeightedFanFlatDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

