import argparse
import os
from os import listdir
from os.path import join
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from math import log10
import datetime
from dataload import DatasetFromFolder_train, DatasetFromFolder_test
from utils import get_scheduler, SSIM, init_net
from network_unet import UNetRes
import warnings
warnings.filterwarnings('ignore')


# Check the number of arguments
if len(sys.argv) != 12:
    print("Usage: python train.py <training_data_path> <model_save_path> <visualization_path> "
          "<pretrained_model_path> <img_size> <batch_size> <epoch_count> <nEpochs> "
          "<generatorLR> <lr_policy> <lr_decay_iters>")
    sys.exit(1)

# Parse command-line arguments
training_path = sys.argv[1]
model_path = sys.argv[2]
graph_path = sys.argv[3]
pretrained_model_path = sys.argv[4]
img_size = int(sys.argv[5])
batch_size = int(sys.argv[6])
epoch_count = int(sys.argv[7])
nEpochs = int(sys.argv[8])
generatorLR = float(sys.argv[9])
lr_policy = sys.argv[10]
lr_decay_iters = int(sys.argv[11])

# Print parameter confirmation
print(f"Training data path: {training_path}")
print(f"Model save path: {model_path}")
print(f"Visualization path: {graph_path}")
print(f"Pretrained model path: {pretrained_model_path}")
print(f"Input image size: {img_size}")
print(f"Batch size: {batch_size}")
print(f"Starting epoch count: {epoch_count}")
print(f"Number of training epochs: {nEpochs}")
print(f"Generator learning rate: {generatorLR}")
print(f"Learning rate policy: {lr_policy}")
print(f"Learning rate decay iterations: {lr_decay_iters}")

# 解析参数
opt = argparse.Namespace(
    training_path=training_path,
    model_path=model_path,
    graph_path=graph_path,
    pretrained_model_path=pretrained_model_path,
    img_size=img_size,
    batch_size=batch_size,
    epoch_count=epoch_count,
    nEpochs=nEpochs,
    generatorLR=generatorLR,
    lr_policy=lr_policy,
    lr_decay_iters=lr_decay_iters
)

print(opt)

try:
    os.makedirs(opt.graph_path)
except OSError:
    pass


if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('-----------------------------Using GPU-----------------------------')

checkpoint = 0

print('===> Loading datasets')
train_set = DatasetFromFolder_train(opt.training_path + '/train', opt.img_size)
test_set = DatasetFromFolder_test(opt.training_path + '/train', opt.img_size)
training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False)

### generator
# Loading SR model
generator = torch.load("{}".format(opt.pretrained_model_path)).to(device)
# for param in generator.m_tail.parameters():
for param in list(generator.parameters())[-2:]:
    param.requires_grad = True

# # generator = UNetRes(in_nc=1, out_nc=1)
# generator = UnetRCAB_ViT2()
# generator = init_net(generator, init_type='normal', init_gain=0.02)

### loss function
L1_loss = nn.L1Loss()
L2_loss = nn.MSELoss()
# adversarial_criterion = nn.BCEWithLogitsLoss()
ssim_criterion = SSIM(window_size=11)

# if gpu is available
if torch.cuda.is_available():
    generator = generator.to(device)
    content_criterion = L2_loss.to(device)
    L1_loss = L1_loss.to(device)
    # adversarial_criterion = adversarial_criterion.to(device)

# learning rate
optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR, betas=(0.9, 0.999))
scheduler_generator = get_scheduler(optim_generator, opt)

# Loading VGG16
vgg16 = models.vgg16(pretrained=True).features.eval()
feature_layer = 30
vgg16 = vgg16[:feature_layer].to(device)
for param in vgg16.parameters():
    param.requires_grad = False

print('--------------------------------------Model training--------------------------------------')
# History
history = pd.DataFrame()
G_loss = []
VAL_mae = []
VAL_mse = []
VAL_nrmse = []
VAL_ssim = []
VAL_psnr = []
start_time = datetime.datetime.now()

best_loss = 100.
for epoch in range(opt.epoch_count, opt.nEpochs):
    mean_pce_loss = 0.0
    mean_l1_loss = 0.0
    mean_l2_loss = 0.0
    mean_ssim_loss = 0.0
    mean_total_loss = 0.0
    count = len(training_data_loader)

    generator.train()
    with torch.set_grad_enabled(True):
        for i, (input1, input2, targets) in enumerate(training_data_loader, 1):
            # low_res_real1 = input1.to(device)
            # low_res_real2 = input2.to(device)
            # high_res_real = targets.to(device)

            # Multi_batches in one batch
            low_res_real1 = input1.squeeze(0).to(device)
            low_res_real2 = input2.squeeze(0).to(device)
            high_res_real = targets.squeeze(0).to(device)

            high_res_fake = generator(low_res_real1)
            generator.zero_grad()

            # --------- calculate loss ---------
            # # Feature loss
            target_features = vgg16(torch.cat([high_res_real] * 3, dim=1))
            fake_features = vgg16(torch.cat([high_res_fake] * 3, dim=1))
            perception_loss = content_criterion(target_features, fake_features)
            mean_pce_loss += perception_loss

            # SSIM loss
            ssim_loss = 1 - ssim_criterion(high_res_real, high_res_fake)
            mean_ssim_loss += ssim_loss

            # # L1 loss
            # generator_l1_loss = 0.5*L1_loss(high_res_real, high_res_fake) + 0.5*L2_loss(high_res_real, high_res_fake)
            # mean_l1_loss += generator_l1_loss

            # # L2 loss
            generator_l2_loss = L2_loss(high_res_fake, high_res_real)
            mean_l2_loss += generator_l2_loss

            # # Total loss
            # if epoch < 500:
            #     generator_total_loss = 0.8*generator_l2_loss + 0.2*ssim_loss + 0.05 * perception_loss
            # else:
            #     generator_total_loss = 1 * generator_l2_loss + 0 * ssim_loss + 0.05 * perception_loss
            generator_total_loss = 0.7 * generator_l2_loss + 0.2 * ssim_loss + 0.05 * perception_loss
            mean_total_loss += generator_total_loss
            generator_total_loss.backward()
            optim_generator.step()

            # elapsed_time = datetime.datetime.now() - start_time
            # print('\r[%d/%d][%d/%d] Generator_Loss (SSIM/Content/Total): %.8f/%.8f/%.8f time: %s'
            #       % (epoch+1, opt.nEpochs, i+1, len(training_data_loader),
            #        ssim_loss.item(), generator_content_loss.item(), generator_total_loss.item(), elapsed_time))

    scheduler_generator.step()

    generator.eval()
    with torch.set_grad_enabled(False):
        G_loss.append(mean_total_loss.detach().cpu().numpy() / count)
        elapsed_time = datetime.datetime.now() - start_time
        print("===> TRAIN: Epoch[{}]:  Loss_G: {:.6f}  Content: {:.6f}  SSIM: {:.6f}  Preception: {:.6f}  time: {}"
              .format(epoch + 1, G_loss[-1], mean_l2_loss / count, 1-(mean_ssim_loss / count), mean_pce_loss / count, elapsed_time))

        mean_mae_loss = 0.0
        mean_mse_loss = 0.0
        mean_nrmse_loss = 0.0
        mean_ssim_loss = 0.0
        mean_psnr_loss = 0.0
        count = len(testing_data_loader)
        record = 0
        for batch_ind, (input1, input2, targets) in enumerate(testing_data_loader):
            low_res_real1 = input1.to(device)
            low_res_real2 = input2.to(device)
            high_res_real = targets.to(device)

            outputs = generator(low_res_real1)

            mae_loss = L1_loss(outputs, high_res_real.detach())
            mean_mae_loss += mae_loss

            mse_loss = content_criterion(outputs, high_res_real.detach())
            mean_mse_loss += mse_loss

            nrmse_loss = np.sqrt(mse_loss.item())
            mean_nrmse_loss += nrmse_loss

            ssim_loss = ssim_criterion(outputs, high_res_real)
            mean_ssim_loss += ssim_loss

            psnr_loss = 10 * log10(1 / mae_loss.item())
            mean_psnr_loss += psnr_loss

            low_res_real1 = low_res_real1.cpu().numpy()
            low_res_real1 = np.transpose(low_res_real1, (0, 2, 3, 1))
            low_res_real2 = low_res_real2.cpu().numpy()
            low_res_real2 = np.transpose(low_res_real2, (0, 2, 3, 1))
            high_res_real = high_res_real.cpu().numpy()
            high_res_real = np.transpose(high_res_real, (0, 2, 3, 1))
            outputs = outputs.cpu().numpy()
            outputs = np.transpose(outputs, (0, 2, 3, 1))
            if batch_ind == record:
                #### ---------- batch size >= 2 ----------
                if not os.path.exists(graph_path):
                    os.makedirs(graph_path)

                gen_imgs = [low_res_real1, low_res_real2, outputs, high_res_real]
                titles = ['Input1', 'Input2', 'Pred', 'SR']
                r, c = 1, 4
                fig, axs = plt.subplots(r, c, figsize=(8, 3))
                for i in range(r):
                    for j in range(c):
                        axs[j].imshow(gen_imgs[j][i], cmap='gray')
                        axs[j].set_title(titles[j])
                        axs[j].axis('off')
                fig.savefig('{}/{}.png'.format(graph_path, epoch + 1))
                # plt.show()
                plt.close()


        VAL_mae.append(mean_mae_loss.detach().cpu().numpy() / count)
        VAL_mse.append(mean_mse_loss.detach().cpu().numpy() / count)
        VAL_nrmse.append(mean_nrmse_loss / count)
        VAL_ssim.append(mean_ssim_loss.detach().cpu().numpy() / count)
        VAL_psnr.append(mean_psnr_loss / count)
        elapsed_time = datetime.datetime.now() - start_time
        print("===> VAL:   Epoch[{}]:  MAE: {:.6f}  MSE: {:.6f}  NRMSE: {:.6f}  SSIM: {:.6f}  PSNR: {:.6f}dB  time: {}".format(
            epoch + 1, VAL_mae[-1], VAL_mse[-1], VAL_nrmse[-1], VAL_ssim[-1], VAL_psnr[-1], elapsed_time))

    epoch_result = dict(epoch=epoch+1, G_loss=round(float(G_loss[-1]), 6),
                        MAE=round(float(VAL_mae[-1]), 6), MSE=round(float(VAL_mse[-1]), 6), NRMSE=round(float(nrmse_loss),6),
                        SSIM=round(float(VAL_ssim[-1]), 6), PSNR=round(float(VAL_psnr[-1]), 6))
    history = history._append(epoch_result, ignore_index=True)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    history_path = "{}/history.csv".format(model_path)
    history.to_csv(history_path, index=False)

    # save the best model
    if (VAL_nrmse[-1]) < best_loss:
        net_g_model_out_path = "{}/netG_epoch_{}.pth".format(model_path, epoch + 1)
        torch.save(generator, net_g_model_out_path)
        best_loss = VAL_nrmse[-1]
        print("Checkpoint saved to {}".format(model_path))

    # Save model at intervals
    if (epoch + 1) % 5 == 0:
        # if not os.path.exists("checkpoints"):
        #     os.mkdir("checkpoints")
        # if not os.path.exists(os.path.join("checkpoints", trainout)):
        #     os.mkdir(os.path.join("checkpoints", trainout))
        # net_g_model_out_path = "checkpoints/{}/netG_epoch_{}.pth".format(trainout, epoch + 1)
        # # net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        # torch.save(generator, net_g_model_out_path)
        # # torch.save(net_d, net_d_model_out_path)
        # print("Checkpoint saved to {}".format("checkpoint" + trainout))

        x_ = range(opt.epoch_count, epoch + 1)
        plt.plot(x_, G_loss, color='r', label='Train Loss')
        plt.plot(x_, VAL_mae, color='b', label='Val Loss')
        plt.legend()
        loss_image_path = "{}/loss.png".format(model_path)
        plt.savefig(loss_image_path)
        plt.close()

print('---------------Training is finished!!!---------------')



