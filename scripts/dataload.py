from os import listdir
from os.path import join
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage import zoom

matplotlib.use('TkAgg')

def augment_img(img, mode=0):
    if (mode == 0) | (mode > 7):
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


class DatasetFromFolder_train(Dataset):
    def __init__(self, image_dir, img_size, num_patches=8):
        super(DatasetFromFolder_train, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b1_path = join(image_dir, "b")
        self.b2_path = join(image_dir, "b")
        self.image_filenames_a = listdir(self.a_path)
        self.image_filenames_a.sort(key=lambda x:str(x[0:-4]))

        self.image_filenames_b1 = listdir(self.b1_path)
        self.image_filenames_b1.sort(key=lambda x:str(x[0:-4]))

        self.image_filenames_b2 = listdir(self.b2_path)
        self.image_filenames_b2.sort(key=lambda x:str(x[0:-4]))
        self.L_size = img_size
        self.num_patches = num_patches

    def __getitem__(self, index):
        a_0 = Image.open(join(self.a_path, self.image_filenames_a[index]))
        a_0 = np.asarray(a_0, dtype=np.float32)

        b1_0 = Image.open(join(self.b1_path, self.image_filenames_b1[index]))
        b1_0 = np.asarray(b1_0, dtype=np.float32)

        b2_0 = Image.open(join(self.b2_path, self.image_filenames_b2[index]))
        b2_0 = np.asarray(b2_0, dtype=np.float32)

        # a_0 = zoom(a_0, zoom=0.5, order=1)
        #     ## ---- SIM data ---
        b1_0 = zoom(b1_0, zoom=2, order=1)
        b2_0 = zoom(b2_0, zoom=2, order=1)

        H, W = b1_0.shape
        sf = 1

        # Initialize lists to store patches
        img_H_list = []
        img_L1_list = []
        img_L2_list = []

        # Extract patches
        for _ in range(self.num_patches):
            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L1 = b1_0[rnd_h: rnd_h + self.L_size, rnd_w: rnd_w + self.L_size]
            img_L2 = b2_0[rnd_h: rnd_h + self.L_size, rnd_w: rnd_w + self.L_size]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
            img_H = a_0[rnd_h_H:rnd_h_H + self.L_size * sf, rnd_w_H:rnd_w_H + self.L_size * sf]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 15)
            img_L1, img_L2, img_H = augment_img(img_L1, mode=mode), augment_img(img_L2, mode=mode), augment_img(img_H, mode=mode)

            # 归一化 1
            img_H = (img_H - np.min(img_H)) / (np.max(img_H) - np.min(img_H) + 1e-8)
            img_L1 = (img_L1 - np.min(img_L1)) / (np.max(img_L1) - np.min(img_L1) + 1e-8)
            img_L2 = (img_L2 - np.min(img_L2)) / (np.max(img_L2) - np.min(img_L2) + 1e-8)

            # Convert to PyTorch tensors
            img_H_list.append(transforms.ToTensor()(img_H))
            img_L1_list.append(transforms.ToTensor()(img_L1))
            img_L2_list.append(transforms.ToTensor()(img_L2))

        # 将像素值转换为PyTorch张量
        # Convert lists to tensors
        img_H_tensors = torch.stack(img_H_list)
        img_L1_tensors = torch.stack(img_L1_list)
        img_L2_tensors = torch.stack(img_L2_list)

        return img_L1_tensors, img_L2_tensors, img_H_tensors

    def __len__(self):
        return len(self.image_filenames_b1)


class DatasetFromFolder_test(Dataset):
    def __init__(self, image_dir, img_size):
        super(DatasetFromFolder_test, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b1_path = join(image_dir, "b")
        self.b2_path = join(image_dir, "b")
        self.image_filenames_a = listdir(self.a_path)
        self.image_filenames_a.sort(key=lambda x:str(x[0:-4]))

        self.image_filenames_b1 = listdir(self.b1_path)
        self.image_filenames_b1.sort(key=lambda x:str(x[0:-4]))

        self.image_filenames_b2 = listdir(self.b2_path)
        self.image_filenames_b2.sort(key=lambda x:str(x[0:-4]))
        self.L_size = img_size

    def __getitem__(self, index):
        a_0 = Image.open(join(self.a_path, self.image_filenames_a[index]))
        a_0 = np.asarray(a_0, dtype=np.float32)

        b1_0 = Image.open(join(self.b1_path, self.image_filenames_b1[index]))
        b1_0 = np.asarray(b1_0, dtype=np.float32)

        b2_0 = Image.open(join(self.b2_path, self.image_filenames_b2[index]))
        b2_0 = np.asarray(b2_0, dtype=np.float32)

        img_H = a_0
        img_L1 = b1_0
        img_L2 = b2_0


        # --------------------------------
        # randomly crop the L patch
        # --------------------------------
        # img_H = zoom(img_H, zoom=0.5, order=1)
             ## ---- SIM data = 2 ---
        img_L1 = zoom(img_L1, zoom=2, order=1)
        img_L2 = zoom(img_L2, zoom=2, order=1)

        H, W = img_L1.shape
        sf = 1

        rnd_h = random.randint(0, max(0, H - self.L_size))
        rnd_w = random.randint(0, max(0, W - self.L_size))
        img_L1 = img_L1[rnd_h: rnd_h + self.L_size, rnd_w: rnd_w + self.L_size]
        img_L2 = img_L2[rnd_h: rnd_h + self.L_size, rnd_w: rnd_w + self.L_size]

        # --------------------------------
        # crop corresponding H patch
        # --------------------------------
        rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
        img_H = img_H[rnd_h_H:rnd_h_H + self.L_size * sf, rnd_w_H:rnd_w_H + self.L_size * sf]

        # # --------------------------------
        # # augmentation - flip and/or rotate
        # # --------------------------------
        # mode = random.randint(0, 15)
        # img_L1, img_L2, img_H = augment_img(img_L1, mode=mode), augment_img(img_L2, mode=mode), augment_img(img_H, mode=mode)


        # 归一化 1
        a_0 = (img_H - np.min(img_H)) / (np.max(img_H) - np.min(img_H) + 1e-8)
        b1_0 = (img_L1 - np.min(img_L1)) / (np.max(img_L1) - np.min(img_L1) + 1e-8)
        b2_0 = (img_L2 - np.min(img_L2)) / (np.max(img_L2) - np.min(img_L2) + 1e-8)

        # 将像素值转换为PyTorch张量s
        a = transforms.ToTensor()(a_0)
        b1 = transforms.ToTensor()(b1_0)
        b2 = transforms.ToTensor()(b2_0)
        return b1, b2, a

    def __len__(self):
        return len(self.image_filenames_b1)
