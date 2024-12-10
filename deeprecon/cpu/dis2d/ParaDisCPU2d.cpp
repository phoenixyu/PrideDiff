#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
#include <atomic>

#define VIEWS_PER_THREAD 128


namespace
{
void atomicAdd(float* ptr, float val) {
    std::atomic<float>* atomic_ptr = reinterpret_cast<std::atomic<float>*>(ptr);
    float old_val = atomic_ptr->load();
    while(!std::atomic_compare_exchange_weak(atomic_ptr, &old_val, old_val + val));
}
}


template<bool TRANSPOSE>
void ProjParaDisCPU2dKernel(
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

    for (int idxView = idxView_start; idxView < idxView_end; idxView++) {
        float view = ViewAngle[idxView];
        float sinVal = sin(view);
        float cosVal = cos(view);
        float Det0Proj;
        if (cosVal * cosVal > 0.5) {
            if (cosVal >= 0) {
                Det0Proj = ((- NumDet / 2.0) * DetSize + BinShift) / cosVal;
            } else {
                Det0Proj = ((NumDet / 2.0) * DetSize + BinShift) / cosVal;
            }

            float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
            float DetInterval = DetSize / fabs(cosVal);
            float PixInterval = ImageSizeX;
            float CoefTemp = (TRANSPOSE)? (DetSize / PixInterval) : (ImageSizeY / DetSize);

            for (int idxrow = 0; idxrow < Height; idxrow++) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = sinVal / cosVal * Pointy + Point0x;
                float Bound0 = fmax(Point0Proj, Det0Proj);
                int idxcol;
                int idxDet;
                if (Point0Proj == Bound0) {
                    idxcol = 0;
                    idxDet = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxDet + 1) * DetInterval + Det0Proj;

                while(idxcol < Width && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
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
                        float coef = (Det1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        Det1Proj += DetInterval;
                    }
                }
            }
        } else {
            if (sinVal >= 0) {
                Det0Proj = ((- NumDet / 2.0) * DetSize + BinShift) / sinVal;
            } else {
                Det0Proj = ((NumDet / 2.0) * DetSize + BinShift) / sinVal;
            }
            float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
            float DetInterval = DetSize / fabs(sinVal);
            float PixInterval = ImageSizeY;
            float CoefTemp = (TRANSPOSE)? (DetSize / PixInterval) : (ImageSizeX / DetSize);

            for (int idxcol = 0; idxcol < Width; idxcol++) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = cosVal / sinVal * Pointx + Point0y;
                float Bound0 = fmax(Point0Proj, Det0Proj);
                int idxrow;
                int idxDet;
                if (Point0Proj == Bound0) {
                    idxrow = 0;
                    idxDet = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxDet + 1) * DetInterval + Det0Proj;

                while(idxrow < Height && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
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
                        float coef = (Det1Proj - Bound0) * CoefTemp;
                        for (int idxBatch = 0; idxBatch < BatchSize; idxBatch++) {
                            float pixVal = Image[(idxBatch * Height + Height - 1 - idxrow) * Width + idxcol];
                            float projTemp = pixVal * coef;
                            int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                            Projection[(idxBatch * NumView + idxView) * NumDet + idxDetTemp] += projTemp;
                        }
                        Bound0 = Det1Proj;
                        idxDet ++;
                        Det1Proj += DetInterval;
                    }
                }
            }            
        }
    }
}


template<bool TRANSPOSE>
void BackProjParaDisCPU2dKernel(
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

    for (int idxView = idxView_start; idxView < idxView_end; idxView++) {
        float view = ViewAngle[idxView];
        float sinVal = sin(view);
        float cosVal = cos(view);
        float Det0Proj;
        if (cosVal * cosVal > 0.5) {
            if (cosVal >= 0) {
                Det0Proj = ((- NumDet / 2.0) * DetSize + BinShift) / cosVal;
            } else {
                Det0Proj = ((NumDet / 2.0) * DetSize + BinShift) / cosVal;
            }

            float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
            float DetInterval = DetSize / fabs(cosVal);
            float PixInterval = ImageSizeX;
            float CoefTemp = (TRANSPOSE)? (ImageSizeY / DetSize) : (DetSize / PixInterval);

            for (int idxrow = 0; idxrow < Height; idxrow++) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = sinVal / cosVal * Pointy + Point0x;
                float Bound0 = fmax(Point0Proj, Det0Proj);
                int idxcol;
                int idxDet;
                if (Point0Proj == Bound0) {
                    idxcol = 0;
                    idxDet = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxDet + 1) * DetInterval + Det0Proj;

                while(idxcol < Width && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
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
                        float coef = (Det1Proj - Bound0) * CoefTemp;
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
                        Det1Proj += DetInterval;
                    }
                }
            }
        } else {
            if (sinVal >= 0) {
                Det0Proj = ((- NumDet / 2.0) * DetSize + BinShift) / sinVal;
            } else {
                Det0Proj = ((NumDet / 2.0) * DetSize + BinShift) / sinVal;
            }
            float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
            float DetInterval = DetSize / fabs(sinVal);
            float PixInterval = ImageSizeY;
            float CoefTemp = (TRANSPOSE)? (ImageSizeX / DetSize) : (DetSize / PixInterval);

            for (int idxcol = 0; idxcol < Width; idxcol++) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = cosVal / sinVal * Pointx + Point0y;
                float Bound0 = fmax(Point0Proj, Det0Proj);
                int idxrow;
                int idxDet;
                if (Point0Proj == Bound0) {
                    idxrow = 0;
                    idxDet = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxDet + 1) * DetInterval + Det0Proj;

                while(idxrow < Height && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
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
                        float coef = (Det1Proj - Bound0) * CoefTemp;
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
                        Det1Proj += DetInterval;
                    }
                }
            }            
        }
    }
}


void ProjParaDisCPU2d(
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
        ProjParaDisCPU2dKernel<false>(Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                ProjParaDisCPU2dKernel<false>, 
                Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}


void ProjTransParaDisCPU2d(
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
        BackProjParaDisCPU2dKernel<true>(Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                BackProjParaDisCPU2dKernel<true>, 
                Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}


void BackProjParaDisCPU2d(
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
        BackProjParaDisCPU2dKernel<false>(Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                BackProjParaDisCPU2dKernel<false>, 
                Image, Projection, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}


void BackProjTransParaDisCPU2d(
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
        ProjParaDisCPU2dKernel<true>(Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
            DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, 0, 1);
    } else {
        std::vector<std::thread> threads(num_threads);
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx] = std::thread(
                ProjParaDisCPU2dKernel<true>, 
                Projection, Image, ViewAngle, BatchSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY, 
                DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift, threadidx, num_threads
            );
        }
        
        for (int threadidx = 0; threadidx < num_threads; threadidx++) {
            threads[threadidx].join();
        }
    }    
}


void GetSysMatrixPara2dKernel(
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

    for (int idxView = 0; idxView < NumView; idxView++) {
        float view = ViewAngle[idxView];
        float sinVal = sin(view);
        float cosVal = cos(view);
        float Det0Proj;
        if (cosVal * cosVal > 0.5) {
            if (cosVal >= 0) {
                Det0Proj = ((- NumDet / 2.0) * DetSize + BinShift) / cosVal;
            } else {
                Det0Proj = ((NumDet / 2.0) * DetSize + BinShift) / cosVal;
            }

            float Point0x = - Width / 2.0 * ImageSizeX + PixXShift;
            float DetInterval = DetSize / fabs(cosVal);
            float PixInterval = ImageSizeX;
            float CoefTemp = ImageSizeY / DetSize;

            for (int idxrow = 0; idxrow < Height; idxrow++) {
                float Pointy = (Height / 2.0 - idxrow - 0.5) * ImageSizeY + PixYShift;
                float Point0Proj = sinVal / cosVal * Pointy + Point0x;
                float Bound0 = fmax(Point0Proj, Det0Proj);
                int idxcol;
                int idxDet;
                if (Point0Proj == Bound0) {
                    idxcol = 0;
                    idxDet = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxcol = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Point1Proj = (idxcol + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxDet + 1) * DetInterval + Det0Proj;

                while(idxcol < Width && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
                        int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                        idxi.push_back(idxView * NumDet + idxDetTemp);
                        idxj.push_back(idxrow * Width + idxcol);
                        value.push_back(coef);
                        Bound0 = Point1Proj;
                        idxcol ++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp;
                        int idxDetTemp = (cosVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                        idxi.push_back(idxView * NumDet + idxDetTemp);
                        idxj.push_back(idxrow * Width + idxcol);
                        value.push_back(coef);
                        Bound0 = Det1Proj;
                        idxDet ++;
                        Det1Proj += DetInterval;
                    }
                }
            }
        } else {
            if (sinVal >= 0) {
                Det0Proj = ((- NumDet / 2.0) * DetSize + BinShift) / sinVal;
            } else {
                Det0Proj = ((NumDet / 2.0) * DetSize + BinShift) / sinVal;
            }
            float Point0y = - Height / 2.0 * ImageSizeY + PixYShift;
            float DetInterval = DetSize / fabs(sinVal);
            float PixInterval = ImageSizeY;
            float CoefTemp = ImageSizeX / DetSize;

            for (int idxcol = 0; idxcol < Width; idxcol++) {
                float Pointx = (idxcol - Width / 2.0 + 0.5) * ImageSizeX + PixXShift;
                float Point0Proj = cosVal / sinVal * Pointx + Point0y;
                float Bound0 = fmax(Point0Proj, Det0Proj);
                int idxrow;
                int idxDet;
                if (Point0Proj == Bound0) {
                    idxrow = 0;
                    idxDet = floor((Bound0 - Det0Proj) / DetInterval);
                } else {
                    idxrow = floor((Bound0 - Point0Proj) / PixInterval);
                    idxDet = 0;
                }
                float Point1Proj = (idxrow + 1) * PixInterval + Point0Proj;
                float Det1Proj = (idxDet + 1) * DetInterval + Det0Proj;

                while(idxrow < Height && idxDet < NumDet) {
                    if (Point1Proj < Det1Proj) {
                        float coef = (Point1Proj - Bound0) * CoefTemp;
                        int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                        idxi.push_back(idxView * NumDet + idxDetTemp);
                        idxj.push_back((Height - 1 - idxrow) * Width + idxcol);
                        value.push_back(coef);
                        Bound0 = Point1Proj;
                        idxrow ++;
                        Point1Proj += PixInterval;
                    } else {
                        float coef = (Det1Proj - Bound0) * CoefTemp;
                        int idxDetTemp = (sinVal >= 0) ? idxDet : (NumDet - 1 - idxDet);
                        idxi.push_back(idxView * NumDet + idxDetTemp);
                        idxj.push_back((Height - 1 - idxrow) * Width + idxcol);
                        value.push_back(coef);
                        Bound0 = Det1Proj;
                        idxDet ++;
                        Det1Proj += DetInterval;
                    }
                }
            }            
        }
    }
}
