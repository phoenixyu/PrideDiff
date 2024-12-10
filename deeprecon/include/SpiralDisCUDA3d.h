#pragma once

void ProjSpiralDisCUDA3d(float *Image, float *Projection, float *ViewAngle, float *SourcePosAxial, float *SourceShiftRadius, float *SourceShiftAngle,
    float *SourceShiftAxial, int BatchSize, int Width, int Height, int Depth, int NumView, int NumDetCol, int NumDetRow, float ImageSizeX,
    float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource, float SourceDetector, float PixXShift,
    float PixYShift, float BinColShift, float BinRowShift);

void ProjTransSpiralDisCUDA3d(float *Image, float *Projection, float *ViewAngle, float *SourcePosAxial, float *SourceShiftRadius, float *SourceShiftAngle,
    float *SourceShiftAxial, int BatchSize, int Width, int Height, int Depth, int NumView, int NumDetCol, int NumDetRow, float ImageSizeX,
    float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource, float SourceDetector, float PixXShift,
    float PixYShift, float BinColShift, float BinRowShift);

void BackProjSpiralDisCUDA3d(float *Image, float *Projection, float *ViewAngle, float *SourcePosAxial, float *SourceShiftRadius, float *SourceShiftAngle,
    float *SourceShiftAxial, int BatchSize, int Width, int Height, int Depth, int NumView, int NumDetCol, int NumDetRow, float ImageSizeX,
    float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource, float SourceDetector, float PixXShift,
    float PixYShift, float BinColShift, float BinRowShift);

void BackProjTransSpiralDisCUDA3d(float *Image, float *Projection, float *ViewAngle, float *SourcePosAxial, float *SourceShiftRadius, float *SourceShiftAngle,
    float *SourceShiftAxial, int BatchSize, int Width, int Height, int Depth, int NumView, int NumDetCol, int NumDetRow, float ImageSizeX,
    float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource, float SourceDetector, float PixXShift,
    float PixYShift, float BinColShift, float BinRowShift);

void BackProjWeightedSpiralDisCUDA3d(float *Image, float *Projection, float *ViewAngle, float *SourcePosAxial, float *SourceShiftRadius, float *SourceShiftAngle,
    float *SourceShiftAxial, int BatchSize, int Width, int Height, int Depth, int NumView, int NumDetCol, int NumDetRow, float ImageSizeX,
    float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource, float SourceDetector, float PixXShift,
    float PixYShift, float BinColShift, float BinRowShift);

void BackProjTransWeightedSpiralDisCUDA3d(float *Image, float *Projection, float *ViewAngle, float *SourcePosAxial, float *SourceShiftRadius, float *SourceShiftAngle,
    float *SourceShiftAxial, int BatchSize, int Width, int Height, int Depth, int NumView, int NumDetCol, int NumDetRow, float ImageSizeX,
    float ImageSizeY, float ImageSizeZ, float DetColSize, float DetRowSize, float IsoSource, float SourceDetector, float PixXShift,
    float PixYShift, float BinColShift, float BinRowShift);