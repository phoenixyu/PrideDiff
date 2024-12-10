#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
#include <atomic>

#define VIEWS_PER_THREAD 128

namespace
{
float Map2x(float sourcex, float sourcey, float pointx, float pointy) {
    return (sourcex * pointy - sourcey * pointx) / (pointy - sourcey);
}

float Map2y(float sourcex, float sourcey, float pointx, float pointy) {
    return (sourcey * pointx - sourcex * pointy) / (pointx - sourcex);
}

float CoordinateWeight(float sourcex, float sourcey, float pointx, float pointy, float r) {
    float d = (sourcex * pointx + sourcey * pointy) / r;
    return r * r / ((r - d) * (r - d));
}

float TriAngCos(float a, float b) {
    return fabs(b) / sqrt(a * a + b * b);
}

void atomicAdd(float* ptr, float val) {
    std::atomic<float>* atomic_ptr = reinterpret_cast<std::atomic<float>*>(ptr);
    float old_val = atomic_ptr->load();
    while(!std::atomic_compare_exchange_weak(atomic_ptr, &old_val, old_val + val));
}
}

void ProjFanFlatDisCPU2dKernel(
    float* Projection,
    const float* Image,
    const float* ViewAngle,
    const int BatchSize,
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
    const float BinShift,
    const int threadidx,
    const int num_threads) {
    
    int num_view_per_thread = (NumView - 1) / num_threads + 1;
    int idxView_start = num_view_per_thread * threadidx;
    int idxView_end = ((idxView_start + num_view_per_thread) > NumView)? NumView : (idxView_start + num_view_per_thread);
    float* DetProj2Axis = (float*)malloc(sizeof(float) * (NumDet + 1));
    float* CoefTemp = (float*)malloc(sizeof(float) * NumDet);

    for (int idxView = idxView_start; idxView < idxView_end; idxView++) {
        float view = ViewAngle[idxView];
        float sinVal = sin(view);
        float cosVal = cos(view);
        float sourcex = - sinVal * IsoSource;
        float sourcey = cosVal * IsoSource;
        float VirDetSize = IsoSource / SourceDetector * DetSize;
        float VirBinShift = IsoSource / SourceDetector * BinShift;
        float Detx;
        float Dety;
        if (cosVal * cosVal > 0.5) {
            float Det0x;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (cosVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0x = Detx;
                DetProj2Axis[idxDet] = Map2x(sourcex, sourcey, Detx, Dety);
            }
            for (int idxDet = 0; idxDet < NumDet; idxDet++) {
                CoefTemp[idxDet] = ImageSizeY / 
                                    (
                                        (DetProj2Axis[idxDet + 1] - DetProj2Axis[idxDet]) *
                                        TriAngCos((DetProj2Axis[idxDet + 1] + DetProj2Axis[idxDet]) / 2.0 - sourcex, sourcey)
                                    );
            }

            float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
            float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
            float DetInterval = VirDetSize * fabs(cosVal);
            for (int idxrow = 0; idxrow < Height; idxrow++) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                int idxcol;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxDet = floor((Bound0Proj2Det - Det0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxcol < Width && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp[idxDet];
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Point1Proj;
                        idxcol ++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp[idxDet];
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        } else {
            float Det0y;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (sinVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0y = Dety;
                DetProj2Axis[idxDet] = Map2y(sourcex, sourcey, Detx, Dety);
            }
            for (int idxDet = 0; idxDet < NumDet; idxDet++) {
                CoefTemp[idxDet] = ImageSizeX / 
                                    (
                                        (DetProj2Axis[idxDet + 1] - DetProj2Axis[idxDet]) *
                                        TriAngCos((DetProj2Axis[idxDet + 1] + DetProj2Axis[idxDet]) / 2.0 - sourcey, sourcex)
                                    );
            }

            float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
            float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
            float DetInterval = VirDetSize * fabs(sinVal);
            for (int idxcol = 0; idxcol < Width; idxcol++) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                int idxrow;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxDet = floor((Bound0Proj2Det - Det0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxrow < Height && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp[idxDet];
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + Height - 1 - idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Point1Proj;
                        idxrow ++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp[idxDet];
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + Height - 1 - idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        }
    }

    free(DetProj2Axis);
    free(CoefTemp);
    DetProj2Axis = NULL;
    CoefTemp = NULL;
}


void ProjTransFanFlatDisCPU2dKernel(
    float* Image,
    const float* Projection,
    const float* ViewAngle,
    const int BatchSize,
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
    const float BinShift,
    const int threadidx,
    const int num_threads) {
    
    int num_view_per_thread = (NumView - 1) / num_threads + 1;
    int idxView_start = num_view_per_thread * threadidx;
    int idxView_end = ((idxView_start + num_view_per_thread) > NumView)? NumView : (idxView_start + num_view_per_thread);
    float* DetProj2Axis = (float*)malloc(sizeof(float) * (NumDet + 1));
    float* CoefTemp = (float*)malloc(sizeof(float) * NumDet);

    for (int idxView = idxView_start; idxView < idxView_end; idxView++) {
        float view = ViewAngle[idxView];
        float sinVal = sin(view);
        float cosVal = cos(view);
        float sourcex = - sinVal * IsoSource;
        float sourcey = cosVal * IsoSource;
        float VirDetSize = IsoSource / SourceDetector * DetSize;
        float VirBinShift = IsoSource / SourceDetector * BinShift;
        float Detx;
        float Dety;
        if (cosVal * cosVal > 0.5) {
            float Det0x;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (cosVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0x = Detx;
                DetProj2Axis[idxDet] = Map2x(sourcex, sourcey, Detx, Dety);
            }
            for (int idxDet = 0; idxDet < NumDet; idxDet++) {
                CoefTemp[idxDet] = ImageSizeY / 
                                    (
                                        (DetProj2Axis[idxDet + 1] - DetProj2Axis[idxDet]) *
                                        TriAngCos((DetProj2Axis[idxDet + 1] + DetProj2Axis[idxDet]) / 2.0 - sourcex, sourcey)
                                    );
            }

            float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
            float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
            float DetInterval = VirDetSize * fabs(cosVal);
            for (int idxrow = 0; idxrow < Height; idxrow++) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                int idxcol;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxDet = floor((Bound0Proj2Det - Det0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxcol < Width && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp[idxDet];
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            float pixVal = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
                            float projTemp = pixVal * coef;
                            if (num_threads == 1) {
                                Image[(idxBatch * Height + idxrow) * Width + idxcol] += projTemp;
                            } else {
                                atomicAdd(Image + (idxBatch * Height + idxrow) * Width + idxcol, projTemp);
                            }
                        }
                        Bound0 = Point1Proj;
                        idxcol ++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp[idxDet];
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            float pixVal = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
                            float projTemp = pixVal * coef;
                            if (num_threads == 1) {
                                Image[(idxBatch * Height + idxrow) * Width + idxcol] += projTemp;
                            } else {
                                atomicAdd(Image + (idxBatch * Height + idxrow) * Width + idxcol, projTemp);
                            }
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        } else {
            float Det0y;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (sinVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0y = Dety;
                DetProj2Axis[idxDet] = Map2y(sourcex, sourcey, Detx, Dety);
            }
            for (int idxDet = 0; idxDet < NumDet; idxDet++) {
                CoefTemp[idxDet] = ImageSizeX / 
                                    (
                                        (DetProj2Axis[idxDet + 1] - DetProj2Axis[idxDet]) *
                                        TriAngCos((DetProj2Axis[idxDet + 1] + DetProj2Axis[idxDet]) / 2.0 - sourcey, sourcex)
                                    );
            }

            float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
            float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
            float DetInterval = VirDetSize * fabs(sinVal);
            for (int idxcol = 0; idxcol < Width; idxcol++) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                int idxrow;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxDet = floor((Bound0Proj2Det - Det0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxrow < Height && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp[idxDet];
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            float pixVal = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
                            float projTemp = pixVal * coef;
                            if (num_threads == 1) {
                                Image[(idxBatch * Height + Height - 1 - idxrow) * Width + idxcol] += projTemp;
                            } else {
                                atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, projTemp);
                            }
                        }
                        Bound0 = Point1Proj;
                        idxrow ++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp[idxDet];
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            float pixVal = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
                            float projTemp = pixVal * coef;
                            if (num_threads == 1) {
                                Image[(idxBatch * Height + Height - 1 - idxrow) * Width + idxcol] += projTemp;
                            } else {
                                atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, projTemp);
                            }
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        }
    }

    free(DetProj2Axis);
    free(CoefTemp);
    DetProj2Axis = NULL;
    CoefTemp = NULL;
}


template<bool FBPWEIGHT>
void BackProjFanFlatDisCPU2dKernel(
    float* Image,
    const float* Projection,
    const float* ViewAngle,
    const int BatchSize,
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
    const float BinShift,
    const int threadidx,
    const int num_threads) {
    
    int num_view_per_thread = (NumView - 1) / num_threads + 1;
    int idxView_start = num_view_per_thread * threadidx;
    int idxView_end = ((idxView_start + num_view_per_thread) > NumView)? NumView : (idxView_start + num_view_per_thread);
    float* DetProj2Axis = (float*)malloc(sizeof(float) * (NumDet + 1));

    for (int idxView = idxView_start; idxView < idxView_end; idxView++) {
        float view = ViewAngle[idxView];
        float sinVal = sin(view);
        float cosVal = cos(view);
        float sourcex = - sinVal * IsoSource;
        float sourcey = cosVal * IsoSource;
        float VirDetSize = IsoSource / SourceDetector * DetSize;
        float VirBinShift = IsoSource / SourceDetector * BinShift;
        float Detx;
        float Dety;
        if (cosVal * cosVal > 0.5) {
            float Det0x;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (cosVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0x = Detx;
                DetProj2Axis[idxDet] = Map2x(sourcex, sourcey, Detx, Dety);
            }

            float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
            float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
            float DetInterval = VirDetSize * fabs(cosVal);
            for (int idxrow = 0; idxrow < Height; idxrow++) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                float CoefTemp = VirDetSize / PixInterval;
                int idxcol;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxDet = floor((Bound0Proj2Det - Det0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Pointx = (idxcol + 0.5) * ImageSizeX + Point0x;
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxcol < Width && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            float pixVal = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
                            float projTemp = pixVal * coef;
                            if (FBPWEIGHT) projTemp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                            if (num_threads == 1) {
                                Image[(idxBatch * Height + idxrow) * Width + idxcol] += projTemp;
                            } else {
                                atomicAdd(Image + (idxBatch * Height + idxrow) * Width + idxcol, projTemp);
                            }
                        }
                        Bound0 = Point1Proj;
                        idxcol ++;
                        Pointx += ImageSizeX;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            float pixVal = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
                            float projTemp = pixVal * coef;
                            if (FBPWEIGHT) projTemp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                            if (num_threads == 1) {
                                Image[(idxBatch * Height + idxrow) * Width + idxcol] += projTemp;
                            } else {
                                atomicAdd(Image + (idxBatch * Height + idxrow) * Width + idxcol, projTemp);
                            }
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        } else {
            float Det0y;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (sinVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0y = Dety;
                DetProj2Axis[idxDet] = Map2y(sourcex, sourcey, Detx, Dety);
            }

            float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
            float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
            float DetInterval = VirDetSize * fabs(sinVal);
            for (int idxcol = 0; idxcol < Width; idxcol++) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                float CoefTemp = VirDetSize / PixInterval;
                int idxrow;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxDet = floor((Bound0Proj2Det - Det0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Pointy = (idxrow + 0.5) * ImageSizeY + Point0y;
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxrow < Height && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            float pixVal = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
                            float projTemp = pixVal * coef;
                            if (FBPWEIGHT) projTemp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                            if (num_threads == 1) {
                                Image[(idxBatch * Height + Height - 1 - idxrow) * Width + idxcol] += projTemp;
                            } else {
                                atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, projTemp);
                            }
                        }
                        Bound0 = Point1Proj;
                        idxrow ++;
                        Pointy += ImageSizeY;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            float pixVal = Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp];
                            float projTemp = pixVal * coef;
                            if (FBPWEIGHT) projTemp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                            if (num_threads == 1) {
                                Image[(idxBatch * Height + Height - 1 - idxrow) * Width + idxcol] += projTemp;
                            } else {
                                atomicAdd(Image + (idxBatch * Height + Height - 1 - idxrow) * Width + idxcol, projTemp);
                            }
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        }
    }

    free(DetProj2Axis);
    DetProj2Axis = NULL;
}


template<bool FBPWEIGHT>
void BackProjTransFanFlatDisCPU2dKernel(
    float* Projection,
    const float* Image,
    const float* ViewAngle,
    const int BatchSize,
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
    const float BinShift,
    const int threadidx,
    const int num_threads) {
    
    int num_view_per_thread = (NumView - 1) / num_threads + 1;
    int idxView_start = num_view_per_thread * threadidx;
    int idxView_end = ((idxView_start + num_view_per_thread) > NumView)? NumView : (idxView_start + num_view_per_thread);
    float* DetProj2Axis = (float*)malloc(sizeof(float) * (NumDet + 1));

    for (int idxView = idxView_start; idxView < idxView_end; idxView++) {
        float view = ViewAngle[idxView];
        float sinVal = sin(view);
        float cosVal = cos(view);
        float sourcex = - sinVal * IsoSource;
        float sourcey = cosVal * IsoSource;
        float VirDetSize = IsoSource / SourceDetector * DetSize;
        float VirBinShift = IsoSource / SourceDetector * BinShift;
        float Detx;
        float Dety;
        if (cosVal * cosVal > 0.5) {
            float Det0x;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (cosVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0x = Detx;
                DetProj2Axis[idxDet] = Map2x(sourcex, sourcey, Detx, Dety);
            }

            float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
            float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
            float DetInterval = VirDetSize * fabs(cosVal);
            for (int idxrow = 0; idxrow < Height; idxrow++) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                float CoefTemp = VirDetSize / PixInterval;
                int idxcol;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxDet = floor((Bound0Proj2Det - Det0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Pointx = (idxcol + 0.5) * ImageSizeX + Point0x;
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxcol < Width && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            if (FBPWEIGHT) projTemp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Point1Proj;
                        idxcol ++;
                        Pointx += ImageSizeX;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            if (FBPWEIGHT) projTemp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        } else {
            float Det0y;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (sinVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0y = Dety;
                DetProj2Axis[idxDet] = Map2y(sourcex, sourcey, Detx, Dety);
            }

            float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
            float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
            float DetInterval = VirDetSize * fabs(sinVal);
            for (int idxcol = 0; idxcol < Width; idxcol++) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                float CoefTemp = VirDetSize / PixInterval;
                int idxrow;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxDet = floor((Bound0Proj2Det - Det0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Pointy = (idxrow + 0.5) * ImageSizeY + Point0y;
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxrow < Height && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + Height - 1 - idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            if (FBPWEIGHT) projTemp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Point1Proj;
                        idxrow ++;
                        Pointy += ImageSizeY;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + Height - 1 - idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            if (FBPWEIGHT) projTemp *= CoordinateWeight(sourcex, sourcey, Pointx, Pointy, IsoSource);
                            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        }
    }

    free(DetProj2Axis);
    DetProj2Axis = NULL;
}



void ProjFanFlatDisCPU2d(
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


    int num_cores = std::thread::hardware_concurrency();
    int num_threads = (NumView - 1) / VIEWS_PER_THREAD + 1;
    num_threads = (num_threads > num_cores)? num_cores : num_threads;

    if (num_threads <= 1) {
        ProjFanFlatDisCPU2dKernel(Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                ProjFanFlatDisCPU2dKernel, 
                Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}

void ProjTransFanFlatDisCPU2d(
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


    int num_cores = std::thread::hardware_concurrency();
    int num_threads = (NumView - 1) / VIEWS_PER_THREAD + 1;
    num_threads = (num_threads > num_cores)? num_cores : num_threads;

    if (num_threads <= 1) {
        ProjTransFanFlatDisCPU2dKernel(Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                ProjTransFanFlatDisCPU2dKernel, 
                Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}

void BackProjFanFlatDisCPU2d(
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


    int num_cores = std::thread::hardware_concurrency();
    int num_threads = (NumView - 1) / VIEWS_PER_THREAD + 1;
    num_threads = (num_threads > num_cores)? num_cores : num_threads;

    if (num_threads <= 1) {
        BackProjFanFlatDisCPU2dKernel<false>(Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                BackProjFanFlatDisCPU2dKernel<false>, 
                Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}


void BackProjTransFanFlatDisCPU2d(
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


    int num_cores = std::thread::hardware_concurrency();
    int num_threads = (NumView - 1) / VIEWS_PER_THREAD + 1;
    num_threads = (num_threads > num_cores)? num_cores : num_threads;

    if (num_threads <= 1) {
        BackProjTransFanFlatDisCPU2dKernel<false>(Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                BackProjTransFanFlatDisCPU2dKernel<false>, 
                Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}


void BackProjWeightedFanFlatDisCPU2d(
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


    int num_cores = std::thread::hardware_concurrency();
    int num_threads = (NumView - 1) / VIEWS_PER_THREAD + 1;
    num_threads = (num_threads > num_cores)? num_cores : num_threads;

    if (num_threads <= 1) {
        BackProjFanFlatDisCPU2dKernel<true>(Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                BackProjFanFlatDisCPU2dKernel<true>, 
                Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}


void BackProjTransWeightedFanFlatDisCPU2d(
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


    int num_cores = std::thread::hardware_concurrency();
    int num_threads = (NumView - 1) / VIEWS_PER_THREAD + 1;
    num_threads = (num_threads > num_cores)? num_cores : num_threads;

    if (num_threads <= 1) {
        BackProjTransFanFlatDisCPU2dKernel<true>(Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                BackProjTransFanFlatDisCPU2dKernel<true>, 
                Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}


void GetSysMatrixFanFlat2dKernel(
    std::vector<int64_t>& idxi,
    std::vector<int64_t>& idxj,
    std::vector<float>& value,
    const float* ViewAngle,
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

    float* DetProj2Axis = (float*)malloc(sizeof(float) * (NumDet + 1));
    float* CoefTemp = (float*)malloc(sizeof(float) * NumDet);

    for (int idxView = 0; idxView < NumView; idxView++) {
        float view = ViewAngle[idxView];
        float sinVal = sin(view);
        float cosVal = cos(view);
        float sourcex = - sinVal * IsoSource;
        float sourcey = cosVal * IsoSource;
        float VirDetSize = IsoSource / SourceDetector * DetSize;
        float VirBinShift = IsoSource / SourceDetector * BinShift;
        float Detx;
        float Dety;
        if (cosVal * cosVal > 0.5) {
            float Det0x;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (cosVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0x = Detx;
                DetProj2Axis[idxDet] = Map2x(sourcex, sourcey, Detx, Dety);
            }
            for (int idxDet = 0; idxDet < NumDet; idxDet++) {
                CoefTemp[idxDet] = ImageSizeY / 
                                    (
                                        (DetProj2Axis[idxDet + 1] - DetProj2Axis[idxDet]) *
                                        TriAngCos((DetProj2Axis[idxDet + 1] + DetProj2Axis[idxDet]) / 2.0 - sourcex, sourcey)
                                    );
            }

            float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
            float Point1x = Width / 2.0 * ImageSizeX + PixXShift;
            float DetInterval = VirDetSize * fabs(cosVal);
            for (int idxrow = 0; idxrow < Height; idxrow++) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = Map2x(sourcex, sourcey, Point0x, Pointy);
                float Point1Proj = Map2x(sourcex, sourcey, Point1x, Pointy);
                float PixInterval = (Point1Proj - Point0Proj) / Width;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                int idxcol;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcey * Bound0 / ((Bound0 - sourcex) * sinVal / cosVal + sourcey);
                    idxcol = 0;
                    idxDet = floor((Bound0Proj2Det - Det0x) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxcol < Width && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp[idxDet];
                        int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                        idxi.push_back(idxView * NumDet + idxDetTemp);
                        idxj.push_back(idxrow * Width + idxcol);
                        value.push_back(coef);
                        Bound0 = Point1Proj;
                        idxcol ++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp[idxDet];
                        int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                        idxi.push_back(idxView * NumDet + idxDetTemp);
                        idxj.push_back(idxrow * Width + idxcol);
                        value.push_back(coef);
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        } else {
            float Det0y;
            for (int idxDet = 0; idxDet <= NumDet; idxDet++) {
                if (sinVal >= 0) {
                    Detx = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((idxDet - NumDet / 2.0) * VirDetSize + VirBinShift) * sinVal;
                } else {
                    Detx = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * cosVal;
                    Dety = ((NumDet / 2.0 - idxDet) * VirDetSize + VirBinShift) * sinVal;
                }
                if (idxDet == 0) Det0y = Dety;
                DetProj2Axis[idxDet] = Map2y(sourcex, sourcey, Detx, Dety);
            }
            for (int idxDet = 0; idxDet < NumDet; idxDet++) {
                CoefTemp[idxDet] = ImageSizeX / 
                                    (
                                        (DetProj2Axis[idxDet + 1] - DetProj2Axis[idxDet]) *
                                        TriAngCos((DetProj2Axis[idxDet + 1] + DetProj2Axis[idxDet]) / 2.0 - sourcey, sourcex)
                                    );
            }

            float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
            float Point1y = Height / 2.0 * ImageSizeY + PixYShift;
            float DetInterval = VirDetSize * fabs(sinVal);
            for (int idxcol = 0; idxcol < Width; idxcol++) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = Map2y(sourcex, sourcey, Pointx, Point0y);
                float Point1Proj = Map2y(sourcex, sourcey, Pointx, Point1y);
                float PixInterval = (Point1Proj - Point0Proj) / Height;
                float Bound0 = fmax(Point0Proj, DetProj2Axis[0]);
                int idxrow;
                int idxDet;
                float Det1Proj;
                if (Point0Proj == Bound0) {
                    float Bound0Proj2Det = sourcex * Bound0 / ((Bound0 - sourcey) * cosVal / sinVal + sourcex);
                    idxrow = 0;
                    idxDet = floor((Bound0Proj2Det - Det0y) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                while(idxrow < Height && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp[idxDet];
                        int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                        idxi.push_back(idxView * NumDet + idxDetTemp);
                        idxj.push_back((Height - 1 - idxrow) * Width + idxcol);
                        value.push_back(coef);
                        Bound0 = Point1Proj;
                        idxrow ++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp[idxDet];
                        int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                        idxi.push_back(idxView * NumDet + idxDetTemp);
                        idxj.push_back((Height - 1 - idxrow) * Width + idxcol);
                        value.push_back(coef);
                        Bound0 = Det1Proj;
                        idxDet ++;
                        if (idxDet < NumDet) Det1Proj = DetProj2Axis[idxDet + 1];
                    }
                }
            }
        }
    }

    free(DetProj2Axis);
    free(CoefTemp);
    DetProj2Axis = NULL;
    CoefTemp = NULL;
}