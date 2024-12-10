
from model.unet import UNet
from model.pridediff import GaussianDiffusion
from model.trainer import Trainer
from data.dataset import MayoDataset
from datetime import datetime

import torchvision
import os
import errno
import shutil
import argparse

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=10, type=int,
                    help="The number of steps the scheduler takes to go from clean image to an isotropic gaussian. This is also the number of steps of diffusion.")
parser.add_argument('--train_steps', default=400000, type=int,
                    help='The number of iterations for training.')
parser.add_argument('--save_folder', default='./results', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='default', type=str,
                    help='The choice of sampling routine for reversing the diffusion process.')
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l2', type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--exp_name', default="LEARN-DIFF_ld25-V2", type=str)


args = parser.parse_args()
print(args)

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


mayodataset = MayoDataset(dataroot='F://LEARN-DIFFUSION//Result//datasets//newmayo', split='train')
mayotestdataset = MayoDataset(dataroot='F://LEARN-DIFFUSION//Result//datasets//newmayo', split='test')

model = UNet(
    in_channels=1, 
    out_channels=1
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 512,
    channels = 1,
    timesteps = args.time_steps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = Trainer(
    diffusion,
    image_size = 512,
    train_batch_size = args.batch_size,
    train_lr = 1e-5,
    train_num_steps = args.train_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = "./results",
    load_path = args.load_path,
    dataset = mayodataset,
    testdataset = mayotestdataset
)

# trainer.load("F:\\LEARN-DIFFUSION\\Result\\learndiff_v2\\pth\\mayo25\\model_best.pt")
# trainer.infer()
trainer.train()