from __future__ import print_function
import argparse
import os
from os import listdir
from os.path import join
import sys
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import datetime
import pandas as pd
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, img_size, sf):
        super(DatasetFromFolder, self).__init__()
        self.b1_path = image_dir

        self.image_filenames_b1 = listdir(self.b1_path)
        self.image_filenames_b1.sort(key=lambda x: str(x[0:-4]))

        self.L_size = img_size
        self.sf = sf

    def __getitem__(self, index):
        b1_0 = Image.open(join(self.b1_path, self.image_filenames_b1[index]))
        b1_0 = np.asarray(b1_0, dtype=np.float32)

        img_L1 = b1_0

        # --------------------------------
        # randomly crop the L patch
        rnd_h = 200         # STED  0 | SMLM  0 | SIM 200 | EM 100
        rnd_w = 200
        # rnd_h = random.randint(0, max(0, H - self.L_size))
        # rnd_w = random.randint(0, max(0, W - self.L_size))
        #     ## ---- SIM & EM data ---
        img_L1 = zoom(img_L1, zoom=2, order=1)

        img_L1 = img_L1[rnd_h: rnd_h + self.L_size, rnd_w: rnd_w + self.L_size]

        # img_L1 = img_L1[:, 1: -1]
        b1_0 = (img_L1 - np.min(img_L1)) / (np.max(img_L1) - np.min(img_L1) + 1e-8)

        b1 = transforms.ToTensor()(b1_0)
        file_name = self.image_filenames_b1[index]
        return b1, file_name

    def __len__(self):
        return len(self.image_filenames_b1)

def main():
    # 1. Parse command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python test.py <input_image_path> <output_image_path> <model_path>")
        sys.exit(1)

    # Testing settings
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    model_path = sys.argv[3]
    print(f"Input image path: {input_image_path}")
    print(f"Output image path: {output_image_path}")
    print(f"Model path: {model_path}")

    Tag = 1
    sf = 1
    img_size = 512


    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('-----------------------------Using GPU-----------------------------')

    # load model
    # model_path = "checkpoints/{}/netG_epoch_{}.pth".format(opt.model, opt.model_ep)
    # model_path = "checkpoints/{}/{}.pth".format(opt.model, opt.model_name)
    generator = torch.load(model_path).to(device)

    # test dir
    print('===> Loading datasets')
    test_set = DatasetFromFolder('{}'.format(input_image_path), img_size, sf)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    save_path1 = '{}/Input'.format(output_image_path)
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    save_path3 = '{}/Output'.format(output_image_path)
    if not os.path.exists(save_path3):
        os.makedirs(save_path3)

    # process the image and save the result
    with torch.no_grad():
        count = len(testing_data_loader)
        start_time = datetime.datetime.now()
        for batch_ind, batch in enumerate(testing_data_loader):
            input1 = batch[0].to(device)
            file_name = batch[1][0]

            if Tag == 0:
                outputs = input1
            else:
                outputs = generator(input1)

            ### plot images
            input1 = input1.cpu().numpy()
            input1 = np.transpose(input1, (0, 2, 3, 1))
            outputs = outputs.cpu().numpy()
            outputs = (np.transpose(outputs, (0, 2, 3, 1)))

            print(f'Processing {file_name}...')
            if Tag == 1:
                in_Img = (input1 - np.min(input1)) / (np.max(input1) - np.min(input1) + 1e-8)
                in_Img = np.asarray(in_Img[0, :, :, 0]*65535, dtype=np.uint16)
                in_Img = Image.fromarray(in_Img, mode='I;16')
                in_Img.save(os.path.join(save_path1, file_name))

                # pred = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs) + 1e-8)
                # prediction = np.asarray(pred[0, :, :, 0]*65535, dtype=np.uint16)
                # prediction = Image.fromarray(prediction, mode='I;16')
                # prediction.save(os.path.join(save_path3, file_name))

                outputs = np.maximum(outputs, 0)
                pred = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs) + 1e-8)
                pixel_values = pred.flatten()
                sorted_pixel_values = np.sort(pixel_values)
                threshold_index = int(0.01 * len(sorted_pixel_values))
                threshold_value = sorted_pixel_values[threshold_index]
                pred = pred - threshold_value
                pred = np.maximum(pred, 0)  # prediction[prediction < 0] = 0

                prediction = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)
                prediction = np.asarray(prediction[0, :, :, 0] * 65535, dtype=np.uint16)
                prediction = Image.fromarray(prediction, mode='I;16')
                prediction.save(os.path.join(save_path3, file_name))

                # plt.imsave(os.path.join(save_path3, file_name)+'f', prediction, cmap='gray')

        elapsed_time = datetime.datetime.now() - start_time
        print("===> time: {}".format(elapsed_time))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Wrong: {str(e)}")
        sys.exit(1)


