#pragma once

void ProjConeArcDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void ProjTransConeArcDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void BackProjConeArcDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void BackProjTransConeArcDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void BackProjWeightedConeArcDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);

void BackProjTransWeightedConeArcDisCUDA3d(float *Image, float *Projection, float *ViewAngle, int BatchSize, int Width, int Height, int Depth, int NumView,
    int NumDetCol, int NumDetRow, float ImageSizeX, float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource,
    float SourceDetector, float PixXShift, float PixYShift, float PixZShift, float BinColShift, float BinRowShift);