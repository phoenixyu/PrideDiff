import glob
import os
from torch.utils.data import Dataset
import numpy as np

class MayoDataset(Dataset):
    def __init__(self, dataroot, img_size=512, split='train'):
        self.img_size = img_size
        self.split = split
        if split=='train':
            self.img_paths = glob.glob(os.path.join(dataroot, 'train', '*_input.npy'))
        else:
            self.img_paths = glob.glob(os.path.join(dataroot, 'test', '*_input.npy'))
        self.data_len = len(self.img_paths)
       
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        cur_img = self.img_paths[index]

        img_NDCT = np.load(cur_img.replace("input","target"))
        img_LDCT = np.load(cur_img)

        img_NDCT = img_NDCT[np.newaxis, :, :]
        img_LDCT = img_LDCT[np.newaxis, :, :]

        img_NDCT = img_NDCT.astype(np.float32)
        img_LDCT = img_LDCT.astype(np.float32)

        img_name = self.img_paths[index].split("\\")[-1].split(".")[0].replace("_input","")

        return {'NDCT': img_NDCT, 'LDCT': img_LDCT, 'Index': index, "NAME": img_name}
