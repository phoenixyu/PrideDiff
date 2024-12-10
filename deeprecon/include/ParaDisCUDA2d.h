#pragma once


void ProjParaDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void ProjTransParaDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjParaDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);

void BackProjTransParaDisCUDA2d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int NumView, int NumDet,
    float ImageSizeX, float ImageSizeY, float DetSize, float IsoSource, float SourceDetector, float PixXShift, float PixYShift, float BinShift);