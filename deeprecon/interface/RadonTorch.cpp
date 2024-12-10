#include <torch/extension.h>
#include "FanFlatDisCUDA2d.h"
#include "FanFlatDisCPU2d.h"
#include "FanArcDisCUDA2d.h"
#include "FanArcDisCPU2d.h"
#include "ParaDisCUDA2d.h"
#include "ParaDisCPU2d.h"
#include "ConeFlatDisCUDA3d.h"
#include "ConeArcDisCUDA3d.h"
#include "SpiralDisCUDA3d.h"

#define CHECK_DEVICE(x, y, z) AT_ASSERTM(x.device() == y.device() && x.device() == z.device(), "Expected all tensors to be on the same device")
#define CHECK_CONTIGUOUS(x, y, z) AT_ASSERTM(x.is_contiguous() && y.is_contiguous() && z.is_contiguous(), "Expected all tensors to be contiguous")
#define CHECK_INPUT(x, y, z) CHECK_DEVICE(x, y, z); CHECK_CONTIGUOUS(x, y, z)


torch::Tensor ProjFanFlatDis2d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDet}, Image.options());
	if (Image.device().is_cuda()) {
		ProjFanFlatDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		Projection.zero_();
		ProjFanFlatDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);
	}
	return Projection;
}


torch::Tensor ProjTransFanFlatDis2d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		ProjTransFanFlatDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);
	} else {
		ProjTransFanFlatDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);
	}
	return Image;
}


torch::Tensor BackProjFanFlatDis2d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjFanFlatDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		BackProjFanFlatDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	}
	return Image;
}


torch::Tensor BackProjTransFanFlatDis2d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDet}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransFanFlatDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		Projection.zero_();
		BackProjTransFanFlatDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);
	}
	return Projection;
}


torch::Tensor BackProjWeightedFanFlatDis2d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjWeightedFanFlatDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);
	} else {
		BackProjWeightedFanFlatDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);
	}
	return Image;
}


torch::Tensor BackProjTransWeightedFanFlatDis2d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDet}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransWeightedFanFlatDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		Projection.zero_();
		BackProjTransWeightedFanFlatDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	}
	return Projection;
}



torch::Tensor ProjFanArcDis2d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDet}, Image.options());
	if (Image.device().is_cuda()) {
		ProjFanArcDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		Projection.zero_();
		ProjFanArcDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	}
	return Projection;
}


torch::Tensor ProjTransFanArcDis2d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		ProjTransFanArcDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);
	} else {
		ProjTransFanArcDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);
	}
	return Image;
}


torch::Tensor BackProjFanArcDis2d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjFanArcDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		BackProjFanArcDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	}
	return Image;
}


torch::Tensor BackProjTransFanArcDis2d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDet}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransFanArcDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		Projection.zero_();
		BackProjTransFanArcDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);
	}
	return Projection;
}


torch::Tensor BackProjWeightedFanArcDis2d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjWeightedFanArcDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		BackProjWeightedFanArcDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);	
	}
	return Image;
}


torch::Tensor BackProjTransWeightedFanArcDis2d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDet}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransWeightedFanArcDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		Projection.zero_();
		BackProjTransWeightedFanArcDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	}
	return Projection;
}



torch::Tensor ProjParaDis2d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDet}, Image.options());
	if (Image.device().is_cuda()) {
		ProjParaDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		Projection.zero_();
		ProjParaDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	}
	return Projection;
}


torch::Tensor ProjTransParaDis2d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		ProjTransParaDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		ProjTransParaDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);	
	}
	return Image;
}


torch::Tensor BackProjParaDis2d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjParaDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		BackProjParaDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	}
	return Image;
}


torch::Tensor BackProjTransParaDis2d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDet}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransParaDisCUDA2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	} else {
		Projection.zero_();
		BackProjTransParaDisCPU2d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, NumView, NumDet, ImageSizeX, ImageSizeY,
			DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift
		);		
	}
	return Projection;
}



torch::Tensor ProjConeFlatDis3d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDetRow, NumDetCol}, Image.options());
	if (Image.device().is_cuda()) {
		ProjConeFlatDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Projection;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor ProjTransConeFlatDis3d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Depth, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		ProjTransConeFlatDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Image;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjConeFlatDis3d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Depth, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjConeFlatDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Image;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjTransConeFlatDis3d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDetRow, NumDetCol}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransConeFlatDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Projection;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjWeightedConeFlatDis3d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Depth, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjWeightedConeFlatDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Image;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjTransWeightedConeFlatDis3d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDetRow, NumDetCol}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransWeightedConeFlatDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Projection;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}



torch::Tensor ProjConeArcDis3d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDetRow, NumDetCol}, Image.options());
	if (Image.device().is_cuda()) {
		ProjConeArcDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Projection;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor ProjTransConeArcDis3d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Depth, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		ProjTransConeArcDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Image;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjConeArcDis3d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Depth, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjConeArcDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Image;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjTransConeArcDis3d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDetRow, NumDetCol}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransConeArcDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Projection;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjWeightedConeArcDis3d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Image = torch::zeros({BatchSize, ChannelSize, Depth, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjWeightedConeArcDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Image;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjTransWeightedConeArcDis3d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float PixZShift = Options[14].item<float>();
    float BinColShift = Options[15].item<float>();
    float BinRowShift = Options[16].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDetRow, NumDetCol}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransWeightedConeArcDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, PixZShift, BinColShift, BinRowShift
		);
		return Projection;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}



torch::Tensor ProjSpiralDis3d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle, torch::Tensor SourcePos) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	CHECK_INPUT(SourcePos, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float BinColShift = Options[14].item<float>();
    float BinRowShift = Options[15].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	float *SourcePosAxial = SourcePos.index({0, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftRadius = SourcePos.index({1, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAngle = SourcePos.index({2, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAxial = SourcePos.index({3, torch::indexing::Slice()}).data_ptr<float>();
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDetRow, NumDetCol}, Image.options());
	if (Image.device().is_cuda()) {
		ProjSpiralDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			SourcePosAxial, SourceShiftRadius, SourceShiftAngle, SourceShiftAxial,
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, BinColShift, BinRowShift
		);
		return Projection;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor ProjTransSpiralDis3d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle, torch::Tensor SourcePos) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	CHECK_INPUT(SourcePos, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float BinColShift = Options[14].item<float>();
    float BinRowShift = Options[15].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	float *SourcePosAxial = SourcePos.index({0, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftRadius = SourcePos.index({1, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAngle = SourcePos.index({2, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAxial = SourcePos.index({3, torch::indexing::Slice()}).data_ptr<float>();
	auto Image = torch::zeros({BatchSize, ChannelSize, Depth, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		ProjTransSpiralDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			SourcePosAxial, SourceShiftRadius, SourceShiftAngle, SourceShiftAxial,
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, BinColShift, BinRowShift
		);
		return Image;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjSpiralDis3d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle, torch::Tensor SourcePos) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	CHECK_INPUT(SourcePos, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float BinColShift = Options[14].item<float>();
    float BinRowShift = Options[15].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	float *SourcePosAxial = SourcePos.index({0, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftRadius = SourcePos.index({1, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAngle = SourcePos.index({2, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAxial = SourcePos.index({3, torch::indexing::Slice()}).data_ptr<float>();
	auto Image = torch::zeros({BatchSize, ChannelSize, Depth, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjSpiralDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			SourcePosAxial, SourceShiftRadius, SourceShiftAngle, SourceShiftAxial,
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, BinColShift, BinRowShift
		);
		return Image;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjTransSpiralDis3d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle, torch::Tensor SourcePos) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	CHECK_INPUT(SourcePos, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float BinColShift = Options[14].item<float>();
    float BinRowShift = Options[15].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	float *SourcePosAxial = SourcePos.index({0, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftRadius = SourcePos.index({1, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAngle = SourcePos.index({2, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAxial = SourcePos.index({3, torch::indexing::Slice()}).data_ptr<float>();
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDetRow, NumDetCol}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransSpiralDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			SourcePosAxial, SourceShiftRadius, SourceShiftAngle, SourceShiftAxial,
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, BinColShift, BinRowShift
		);
		return Projection;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjWeightedSpiralDis3d(torch::Tensor Projection, torch::Tensor Options, torch::Tensor ViewAngle, torch::Tensor SourcePos) {
  	CHECK_INPUT(Projection, Options, ViewAngle);
	CHECK_INPUT(SourcePos, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float BinColShift = Options[14].item<float>();
    float BinRowShift = Options[15].item<float>();
	int BatchSize = static_cast<int>(Projection.size(0));
    int ChannelSize = static_cast<int>(Projection.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	float *SourcePosAxial = SourcePos.index({0, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftRadius = SourcePos.index({1, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAngle = SourcePos.index({2, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAxial = SourcePos.index({3, torch::indexing::Slice()}).data_ptr<float>();
	auto Image = torch::zeros({BatchSize, ChannelSize, Depth, Height, Width}, Projection.options());
	if (Projection.device().is_cuda()) {
		BackProjWeightedSpiralDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			SourcePosAxial, SourceShiftRadius, SourceShiftAngle, SourceShiftAxial,
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, BinColShift, BinRowShift
		);
		return Image;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor BackProjTransWeightedSpiralDis3d(torch::Tensor Image, torch::Tensor Options, torch::Tensor ViewAngle, torch::Tensor SourcePos) {
  	CHECK_INPUT(Image, Options, ViewAngle);
	CHECK_INPUT(SourcePos, Options, ViewAngle);
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int Depth = Options[2].item<int>();
    int NumDetCol = Options[3].item<int>();
    int NumDetRow = Options[4].item<int>();
    float ImageSizeX = Options[5].item<float>();
    float ImageSizeY = Options[6].item<float>();
    float ImageSizeZ = Options[7].item<float>();
    float DetColSize = Options[8].item<float>();
    float DetRowSize = Options[9].item<float>();
    float IsoSource = Options[10].item<float>();
    float SourceDetector = Options[11].item<float>();
    float PixXShift = Options[12].item<float>();
    float PixYShift = Options[13].item<float>();
    float BinColShift = Options[14].item<float>();
    float BinRowShift = Options[15].item<float>();
	int BatchSize = static_cast<int>(Image.size(0));
    int ChannelSize = static_cast<int>(Image.size(1));
	int NumView = static_cast<int>(ViewAngle.size(0));
	float *SourcePosAxial = SourcePos.index({0, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftRadius = SourcePos.index({1, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAngle = SourcePos.index({2, torch::indexing::Slice()}).data_ptr<float>();
	float *SourceShiftAxial = SourcePos.index({3, torch::indexing::Slice()}).data_ptr<float>();
	auto Projection = torch::empty({BatchSize, ChannelSize, NumView, NumDetRow, NumDetCol}, Image.options());
	if (Image.device().is_cuda()) {
		BackProjTransWeightedSpiralDisCUDA3d(
			Image.data_ptr<float>(), Projection.data_ptr<float>(), ViewAngle.data_ptr<float>(),
			SourcePosAxial, SourceShiftRadius, SourceShiftAngle, SourceShiftAxial,
			BatchSize * ChannelSize, Width, Height, Depth, NumView, NumDetCol, NumDetRow,
			ImageSizeX, ImageSizeY, ImageSizeZ, DetColSize, DetRowSize, IsoSource,
    		SourceDetector, PixXShift, PixYShift, BinColShift, BinRowShift
		);
		return Projection;
	} else {
		AT_ERROR("CPU is not supported yet");
	}
}


torch::Tensor GetSysMatrixFanFlat2d(torch::Tensor Options, torch::Tensor ViewAngle) {
	if (ViewAngle.device().is_cuda() || Options.device().is_cuda()) {
		AT_ERROR("CUDA is not supported");
	}
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int NumView = static_cast<int>(ViewAngle.size(0));

	std::vector<int64_t> idxi(0);
    std::vector<int64_t> idxj(0);
    std::vector<float> value(0);
	GetSysMatrixFanFlat2dKernel(idxi, idxj, value, ViewAngle.data_ptr<float>(), Width, Height, NumView, NumDet, 
		ImageSizeX, ImageSizeY, DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift);
	
	int64_t M = (int64_t)NumView * (int64_t)NumDet;
	int64_t N = (int64_t)Width * (int64_t)Height;
	auto indices_i = torch::from_blob(idxi.data(), {static_cast<int64_t>(idxi.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
	auto indices_j = torch::from_blob(idxj.data(), {static_cast<int64_t>(idxj.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
	auto values = torch::from_blob(value.data(), {static_cast<int64_t>(value.size())}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
	auto indices = torch::stack({indices_i, indices_j}).view({2, -1});
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kSparse);
	auto matrix = torch::sparse_coo_tensor(indices, values, {M, N}, options);
	return matrix;
}


torch::Tensor GetSysMatrixFanArc2d(torch::Tensor Options, torch::Tensor ViewAngle) {
	if (ViewAngle.device().is_cuda() || Options.device().is_cuda()) {
		AT_ERROR("CUDA is not supported");
	}
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int NumView = static_cast<int>(ViewAngle.size(0));

	std::vector<int64_t> idxi(0);
    std::vector<int64_t> idxj(0);
    std::vector<float> value(0);
	GetSysMatrixFanArc2dKernel(idxi, idxj, value, ViewAngle.data_ptr<float>(), Width, Height, NumView, NumDet, 
		ImageSizeX, ImageSizeY, DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift);
	
	int64_t M = (int64_t)NumView * (int64_t)NumDet;
	int64_t N = (int64_t)Width * (int64_t)Height;
	auto indices_i = torch::from_blob(idxi.data(), {static_cast<int64_t>(idxi.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
	auto indices_j = torch::from_blob(idxj.data(), {static_cast<int64_t>(idxj.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
	auto values = torch::from_blob(value.data(), {static_cast<int64_t>(value.size())}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
	auto indices = torch::stack({indices_i, indices_j}).view({2, -1});
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kSparse);
	auto matrix = torch::sparse_coo_tensor(indices, values, {M, N}, options);
	return matrix;
}


torch::Tensor GetSysMatrixPara2d(torch::Tensor Options, torch::Tensor ViewAngle) {
	if (ViewAngle.device().is_cuda() || Options.device().is_cuda()) {
		AT_ERROR("CUDA is not supported");
	}
	int Width = Options[0].item<int>();
    int Height = Options[1].item<int>();
    int NumDet = Options[2].item<int>();
    float ImageSizeX = Options[3].item<float>();
    float ImageSizeY = Options[4].item<float>();
    float DetSize = Options[5].item<float>();
    float IsoSource = Options[6].item<float>();
    float SourceDetector = Options[7].item<float>();
    float PixXShift = Options[8].item<float>();
    float PixYShift = Options[9].item<float>();
    float BinShift = Options[10].item<float>();
	int NumView = static_cast<int>(ViewAngle.size(0));

	std::vector<int64_t> idxi(0);
    std::vector<int64_t> idxj(0);
    std::vector<float> value(0);
	GetSysMatrixPara2dKernel(idxi, idxj, value, ViewAngle.data_ptr<float>(), Width, Height, NumView, NumDet, 
		ImageSizeX, ImageSizeY, DetSize, IsoSource, SourceDetector, PixXShift, PixYShift, BinShift);
	
	int64_t M = (int64_t)NumView * (int64_t)NumDet;
	int64_t N = (int64_t)Width * (int64_t)Height;
	auto indices_i = torch::from_blob(idxi.data(), {static_cast<int64_t>(idxi.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
	auto indices_j = torch::from_blob(idxj.data(), {static_cast<int64_t>(idxj.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
	auto values = torch::from_blob(value.data(), {static_cast<int64_t>(value.size())}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
	auto indices = torch::stack({indices_i, indices_j}).view({2, -1});
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kSparse);
	auto matrix = torch::sparse_coo_tensor(indices, values, {M, N}, options);
	return matrix;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("ProjFanFlatDis2d", &ProjFanFlatDis2d);
	m.def("ProjTransFanFlatDis2d", &ProjTransFanFlatDis2d);
	m.def("BackProjFanFlatDis2d", &BackProjFanFlatDis2d);
	m.def("BackProjTransFanFlatDis2d", &BackProjTransFanFlatDis2d);
	m.def("BackProjWeightedFanFlatDis2d", &BackProjWeightedFanFlatDis2d);
	m.def("BackProjTransWeightedFanFlatDis2d", &BackProjTransWeightedFanFlatDis2d);

	m.def("ProjFanArcDis2d", &ProjFanArcDis2d);
	m.def("ProjTransFanArcDis2d", &ProjTransFanArcDis2d);
	m.def("BackProjFanArcDis2d", &BackProjFanArcDis2d);
	m.def("BackProjTransFanArcDis2d", &BackProjTransFanArcDis2d);
	m.def("BackProjWeightedFanArcDis2d", &BackProjWeightedFanArcDis2d);
	m.def("BackProjTransWeightedFanArcDis2d", &BackProjTransWeightedFanArcDis2d);

	m.def("ProjParaDis2d", &ProjParaDis2d);
	m.def("ProjTransParaDis2d", &ProjTransParaDis2d);
	m.def("BackProjParaDis2d", &BackProjParaDis2d);
	m.def("BackProjTransParaDis2d", &BackProjTransParaDis2d);

	m.def("ProjConeFlatDis3d", &ProjConeFlatDis3d);
	m.def("ProjTransConeFlatDis3d", &ProjTransConeFlatDis3d);
	m.def("BackProjConeFlatDis3d", &BackProjConeFlatDis3d);
	m.def("BackProjTransConeFlatDis3d", &BackProjTransConeFlatDis3d);
	m.def("BackProjWeightedConeFlatDis3d", &BackProjWeightedConeFlatDis3d);
	m.def("BackProjTransWeightedConeFlatDis3d", &BackProjTransWeightedConeFlatDis3d);

	m.def("ProjConeArcDis3d", &ProjConeArcDis3d);
	m.def("ProjTransConeArcDis3d", &ProjTransConeArcDis3d);
	m.def("BackProjConeArcDis3d", &BackProjConeArcDis3d);
	m.def("BackProjTransConeArcDis3d", &BackProjTransConeArcDis3d);
	m.def("BackProjWeightedConeArcDis3d", &BackProjWeightedConeArcDis3d);
	m.def("BackProjTransWeightedConeArcDis3d", &BackProjTransWeightedConeArcDis3d);

	m.def("ProjSpiralDis3d", &ProjSpiralDis3d);
	m.def("ProjTransSpiralDis3d", &ProjTransSpiralDis3d);
	m.def("BackProjSpiralDis3d", &BackProjSpiralDis3d);
	m.def("BackProjTransSpiralDis3d", &BackProjTransSpiralDis3d);
	m.def("BackProjWeightedSpiralDis3d", &BackProjWeightedSpiralDis3d);
	m.def("BackProjTransWeightedSpiralDis3d", &BackProjTransWeightedSpiralDis3d);

	m.def("GetSysMatrixFanFlat2d", &GetSysMatrixFanFlat2d);
	m.def("GetSysMatrixFanArc2d", &GetSysMatrixFanArc2d);
	m.def("GetSysMatrixPara2d", &GetSysMatrixPara2d);

}