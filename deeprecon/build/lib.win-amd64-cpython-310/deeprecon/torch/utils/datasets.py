import os
import glob
import torch
import numpy as np
import pydicom as pd
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Resize

class MayoData(Dataset):
    def __init__(self, root, split="train", split_ratio="8,0,2", transform=None, size=512, use_patch=False, patch_size=64) -> None:
        super().__init__()
        ratio = []
        for r in split_ratio.split(","):
            ratio.append(int(r))

        if split == "train":
            idx = [0, ratio[0]]
        elif split == "test":
            idx = [sum(ratio[:2]), sum(ratio)]
        elif split == "vali" :
            if not ratio[1] == 0:
                idx = [ratio[0], sum(ratio[:2])]
            else:
                idx = [sum(ratio[:2]), sum(ratio)]
        else:
            raise NotImplementedError(f"Wrong split {split}.")
        
        patient_nd_path = glob.glob(os.path.join(root, "full*"))[0]        
        patient_ld_path = glob.glob(os.path.join(root, "quarter*"))[0]
        patient_list = os.listdir(patient_nd_path)

        n = len(patient_list)
        nd = n // sum(ratio)
        patient_list = patient_list[idx[0] * nd : idx[1] * nd]

        ndct_list = []
        ldct_list = []
        for patient in patient_list:
            _ndct_list = glob.glob(os.path.join(patient_nd_path, patient, "*", "*.IMA"))
            _ldct_list = glob.glob(os.path.join(patient_ld_path, patient, "*", "*.IMA"))
            _ndct_list.sort()
            _ldct_list.sort()

            ndct_list = ndct_list + _ndct_list
            ldct_list = ldct_list + _ldct_list

        self.ndct_list = np.array(ndct_list)
        self.ldct_list = np.array(ldct_list)

        assert len(self.ndct_list) == len(self.ldct_list), "Mismatching number of full dose CT and low dose CT"
        self.data_len = len(self.ndct_list)
        self.split = split
        self.transform = transform
        self.size = size
        self.use_patch = use_patch
        self.patch_size = tuple((patch_size, patch_size)) if np.isscalar(patch_size) else tuple(patch_size)
        self.resize = None if size == 512 else Resize(size)


    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        ndct = pd.dcmread(self.ndct_list[index]).pixel_array
        ldct = pd.dcmread(self.ldct_list[index]).pixel_array
        ndct = ndct.astype(np.float32)
        ldct = ldct.astype(np.float32)
        if self.transform is not None:
            ndct = self.transform(ndct)
            ldct = self.transform(ldct)
            return ndct, ldct
        ndct = torch.from_numpy(ndct).to(torch.float32) / 2000
        ldct = torch.from_numpy(ldct).to(torch.float32) / 2000
        ndct = ndct[None, ...]
        ldct = ldct[None, ...]
        if self.resize is not None:
            ndct = self.resize(ndct)
            ldct = self.resize(ldct)
        ndct.clamp_(0., 1.)
        ldct.clamp_(0., 1.)

        if not self.use_patch:
            return ndct, ldct
        else:
            ndct_patch, ldct_patch = _extract_patch(ndct, ldct, self.patch_size)
            return ndct_patch, ldct_patch

def _extract_patch(img1, img2, patch_size):
    m, n = img1.shape[-2:]
    ratio = 0.
    while ratio < 0.2:
        idxi = torch.randint(0, m - patch_size[0] + 1, [])
        idxj = torch.randint(0, n - patch_size[1] + 1, [])
        img1_patch = img1[..., idxi:idxi+patch_size[0], idxj:idxj+patch_size[1]]
        img2_patch = img2[..., idxi:idxi+patch_size[0], idxj:idxj+patch_size[1]]
        ratio = (img1_patch > 0.1).sum() / np.prod(patch_size)
    return img1_patch, img2_patch