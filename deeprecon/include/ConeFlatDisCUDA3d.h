#pragma once


void ProjConeFlatDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void ProjTransConeFlatDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void BackProjConeFlatDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void BackProjTransConeFlatDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void BackProjWeightedConeFlatDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void BackProjTransWeightedConeFlatDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);