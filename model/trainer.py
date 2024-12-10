import torch
import copy
import numpy as np
import math
import cv2
import os
import tqdm

from pathlib import Path
from torch.optim import Adam
from torch.utils import data
from functools import partial
from torchvision import utils
from torchvision.utils import make_grid
from scipy.io import savemat

from utils.measure import *

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

def cycle(dl):
    while True:
        for data in dl:
            yield data

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def tensor2img(tensor, out_type=np.uint8, min_max=(0.21, 0.31)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def tensor_clamp(tensor, min_max=(0.21, 0.31)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  
    return tensor

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        *,
        ema_decay = 0.995,
        image_size = 256,
        train_batch_size = 1,
        train_lr = 2e-5,
        train_num_steps = 1000000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 2000,
        results_folder = './results',
        load_path = None,
        dataset = None,
        testdataset = None,
        shuffle = False
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = dataset
        self.test_dataset = testdataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=0, drop_last=True))
        self.test_loader = data.DataLoader(self.test_dataset, batch_size = 1, shuffle=False, pin_memory=True, num_workers=0, drop_last=True)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.fp16 = fp16

        self.best_psnr = 0.0

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None, is_best=False, psnr=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

        if is_best:
            torch.save(data, str(self.results_folder / f'model_best_{psnr}.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    def add_title(self, path, title):

        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)

    def train(self):

        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                cur_data = next(self.dl)
                data_1 = cur_data["NDCT"]
                data_2 = cur_data["LDCT"]
                
                data_1, data_2 = data_1.cuda(), data_2.cuda()
                loss = torch.mean(self.model(data_1, data_2))
                if self.step % 100 == 0:
                    print(f'{self.step}: {loss.item()}')
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            acc_loss = acc_loss + (u_loss/self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            self.step += 1

            if self.step % self.save_and_sample_every == 0:
                self.test()
                self.ema_model.train()

        print('training completed')

    @torch.no_grad()
    def infer(self):
        import time
        for data in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose, name = data["LDCT"].cuda(), data["NDCT"].cuda(), data["NAME"][0]
            batches,_,_,_ = low_dose.shape
            
            xt, direct_recons, recon_images = self.ema_model.module.sample(batch_size=batches, img=low_dose)
            
            ldct_img = low_dose.detach().float().cpu().numpy().squeeze()
            ndct_img = full_dose.detach().float().cpu().numpy().squeeze()
            # out_img = direct_recons.detach().float().cpu().numpy().squeeze()
            out_img = recon_images.detach().float().cpu().numpy().squeeze()

            savemat("F:\\LEARN-DIFFUSION\\Result\\learndiff_v2\\mayo25\\{}.mat".format(name), {"ldct":ldct_img, "ndct":ndct_img, "denoise":out_img})
            # savemat("F:\\LEARN-DIFFUSION\\Result\\learndiff\\mayo_sv_120\\{}.mat".format(name), {"ldct":ldct_img, "ndct":ndct_img, "denoise":out_img, "tmp":tmp_img, "tmp2":tmp_img2, "tmp3":tmp_img3, "tmp4":tmp_img4, "tmp5":tmp_img5})
            # print("{}, Save complete.".format(name))

    @torch.no_grad()
    def test(self):
        self.ema_model.eval()

        if self.step != 0 and self.step % self.save_and_sample_every == 0:
            psnr, ssim, rmse = 0., 0., 0.
            for data in tqdm.tqdm(self.test_loader, desc='test'):
                low_dose, full_dose = data["LDCT"].cuda(), data["NDCT"].cuda()
                batches,_,_,_ = low_dose.shape
                xt, direct_recons, recon_images = self.ema_model.module.sample(batch_size=batches, img=low_dose)
                data_range = full_dose.max() - full_dose.min()
                psnr_score, ssim_score, rmse_score = compute_measure(full_dose, recon_images, data_range)
                psnr += psnr_score / len(self.test_loader)
                ssim += ssim_score / len(self.test_loader)
                rmse += rmse_score / len(self.test_loader)

            print("Step:{} full test result. PSNR:{}, SSIM:{}, RMSE:{}".format(self.step, psnr, ssim, rmse))

            self.gen_test_data()

            self.save(self.step)
            if psnr > self.best_psnr:
                self.best_psnr = psnr
                self.save(is_best=True, psnr=self.best_psnr)


    @torch.no_grad()
    def gen_test_data(self):
        test_images = [self.test_dataset[i] for i in range(0, min(320, len(self.test_dataset)), 40)]
        low_dose = torch.stack([torch.from_numpy(x["LDCT"]) for x in test_images], dim=0).cuda()
        full_dose = torch.stack([torch.from_numpy(x["NDCT"]) for x in test_images], dim=0).cuda()
        batches,_,_,_ = low_dose.shape
        xt, direct_recons, recon_images = self.ema_model.module.sample(batch_size=batches, img=low_dose)

        utils.save_image(tensor_clamp(low_dose), str(self.results_folder / f'sample-ndct.png'), nrow=4)
        utils.save_image(tensor_clamp(full_dose), str(self.results_folder / f'sample-ldct.png'), nrow=4)
        utils.save_image(tensor_clamp(recon_images), str(self.results_folder / f'sample-recon-{self.step}.png'), nrow = 4)
        utils.save_image(tensor_clamp(direct_recons), str(self.results_folder / f'sample-direct_recons-{self.step}.png'), nrow=4)

        data_range = full_dose.max() - full_dose.min()
        psnr_score, ssim_score, rmse_score = compute_measure(full_dose, recon_images, data_range)

        print("Step:{} tiny test result. PSNR:{}, SSIM:{}, RMSE:{}".format(self.step, psnr_score, ssim_score, rmse_score))
   