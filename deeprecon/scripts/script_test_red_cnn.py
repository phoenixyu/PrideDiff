import argparse
import torch
import torchmetrics
import numpy as np
import pytorch_lightning as pl
import deeprecon.torch.model as model
from torch.utils.data import DataLoader
from deeprecon.torch.utils.datasets import MayoData
from deeprecon.torch.utils.utils import (
    create_projector2d,
    add_dict_to_argparser,
    args_to_dict,
)

def model_defaults():
    """
    Defaults for model.
    """
    return dict(

        # in_channel=1,
        # out_channel=1,
        # hid_channel=48,
        # kernel_size=5,
        # padding=2,
    )


def geometry_defaults():
    """
    Defaults for geometry.
    """
    return dict(
        image_size=512,
        num_view=64,
        num_det=736,
        pix_size=0.6641,
        det_size=1.2858,
        iso_source=595,
        source_detector=1085.6,
        pixshift=0.,
        binshift=0.,
        scan_type="flat",
        method="dis",
        filter_type="ramp",
        trainable=False,
        views=None,
    )

def data_defaults():
    """
    Defaults for mayo dataset.
    """
    return dict(
        root="D:/Dataset/Mayo_Grand_Chanllenge/Patient_Data/Training_Image_Data/1mm B30",
        split_ratio="8,0,2",
        transform=None,
        size=512,
        use_patch=False,
        patch_size=64,
    )


def training_defaults():
    """
    Defaults for model training.
    """
    return dict(
        model="red_cnn",
        num_device=1,
        num_loader_workers=4,
        batch_size=32,
        epochs=20000,
        lr=1e-4,
        weight_decay=0.0,
        scheduler_step=2000,
        scheduler_gamma=0.5,
        log_interval=100,
        use_fp16=False,
        resume_checkpoint="",
        is_save_res=False,
        res_dir="result",
    )

def create_argparser():
    defaults = training_defaults()
    defaults.update(model_defaults())
    defaults.update(geometry_defaults())
    defaults.update(data_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


class net(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = eval("model." + kwargs["model"].lower())(**kwargs)
        self.batch_size = kwargs["batch_size"]
        self.is_save_res = kwargs["is_save_res"]
        self.res_dir = kwargs["res_dir"]
        self.psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=[0., 1.])
        self.ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=[0., 1.])
    
    def test_step(self, batch, batch_idx):
        ndct, ldct = batch
        preds = self.model.test(x=ldct)
        psnr = self.psnr_metric(preds, ndct)
        ssim = self.ssim_metric(preds, ndct)
        self.log_dict({"psnr": psnr, "ssim": ssim}, on_step=True)
        if self.is_save_res:
            self.save_res(preds, ndct, batch_idx)

    def test_step_end(self, *args, **kwargs):
        self.log_dict({"psnr":self.psnr_metric.compute(), "ssim":self.ssim_metric.compute()})

    def save_res(self, preds, targets, batch_idx):
        for i in range(preds.shape[0]):
            pred = preds[i].to("cpu").numpy().squeeze()
            target = targets[i].to("cpu").numpy().squeeze()
            np.savez("{}/res_{:4d}.npz".format(self.res_dir, batch_idx * self.batch_size + i), {"pred": pred, "target":target})

def main():
    args = create_argparser().parse_args()
    network = net(**args_to_dict(args, args.keys()))
    network.load_from_checkpoint(args.resume_checkpoint)
    precision = 32 if not args.use_fp16 else 16
    trainer = pl.Trainer(devices=args.num_device, log_every_n_steps=args.log_interval, max_epochs=args.epochs, precision=precision, strategy='ddp')
    dataset = MayoData(root=args.root, split="test", split_ratio=args.split_ratio, 
                    transform=args.transform, size=args.size, use_patch=args.use_patch, patch_size=args.patch_size)
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_loader_workers)
    trainer.test(network, data)


if __name__ == "__main__":
    main()
