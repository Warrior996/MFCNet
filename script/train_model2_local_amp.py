# encoding: utf-8
import re
import sys
import os
from tabnanny import verbose
import csv
import time
import numpy as np
import logging
from tqdm import tqdm

import torch
from torch._C import device
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset.build_data import ChestXrayDataSet
from model.build_local_model import build_model
from model.UNet import UNet
from utils.net_utils import Attention_gen_patchs, compute_AUCs
from utils.build_optimizer_lr_scheduler import optimizer_lr_scheduler
from main import arg_parse
args = arg_parse()

# np.set_printoptions(threshold = np.nan)
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def train(args):

    logging.info("[Info]: Loading Data ...")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=args.dataset_dir,
                                     image_list_file=args.trainSet,
                                     transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                     ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_works, pin_memory=True)
    
    val_dataset = ChestXrayDataSet(data_dir=args.dataset_dir,
                                   image_list_file=args.valSet,
                                   transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_works, pin_memory=True)

    logging.info("[Info]: Data has been loaded ...")

    logging.info("[Info]: Loading Model ...")

    # initialize and load the model
    Local_Branch_model = build_model(args)
    logging.info("Local_Branch_model:{}".format(Local_Branch_model))

    model_unet = UNet(n_channels=3, n_classes=1).cuda()     # initialize model
    checkpoint = torch.load(args.UNET_PATH)
    model_unet.load_state_dict(checkpoint)                  # strict=False
    logging.info("=> loaded well-trained unet model checkpoint: {}".format(args.UNET_PATH))
    model_unet.eval()

    torch.backends.cudnn.benchmark = True
    criterion = nn.BCEWithLogitsLoss()
    optimizer_local, lr_scheduler_local = optimizer_lr_scheduler(Local_Branch_model, args.local_lr)
    logging.info("[Info]: Model has been loaded ...")

    logging.info("[Info]: Starting training ...")
    epoch_losses_local_train = []
    epoch_losses_local_val = []
    best_loss = 999999
    AUROC_max = 0
    scaler = GradScaler()
    for epoch in range(args.num_epochs):
        since = time.time()
        logging.info('Epoch: {}/{}'.format(epoch, args.num_epochs - 1))
        logging.info('-' * 10)
        # set the mode of model
        Local_Branch_model.train()  # set model to training mode
        running_loss_local = 0.0
        # Iterate over data
        train_loader = tqdm(train_loader, file=sys.stdout, ncols=60)
        for i, (input, target) in enumerate(train_loader):
            input_var, target_var = input.cuda(), target.cuda()
            optimizer_local.zero_grad(set_to_none=True)
            with autocast():
                var_mask = model_unet(input_var)
                output_local, _, pool_local = Local_Branch_model(input_var, var_mask)
                loss = criterion(output_local, target_var)
            torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            scaler.step(optimizer_local)
            scaler.update()
            running_loss_local += loss.data.item()

        epoch_loss_train = float(running_loss_local) / float(i)
        epoch_losses_local_train.append(epoch_loss_train)
        logging.info("Local_Train_losses: {}".format(epoch_losses_local_train))

        logging.info("[Info]: Starting valing ...")
        gt = torch.FloatTensor().cuda()
        pred_local = torch.FloatTensor().cuda()
        # switch to evaluate mode
        Local_Branch_model.eval()
        running_loss_local = 0.0
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=60)
        for i, (inp, target) in enumerate(val_loader):
            with torch.no_grad():
                input_var, target = inp.cuda(), target.cuda()
                with autocast():
                    var_mask = model_unet(input_var)
                    output_local, _, pool_local = Local_Branch_model(input_var, var_mask)
                    loss_local = criterion(output_local, target)
                running_loss_local += loss_local.data.item()
                gt = torch.cat((gt, target), 0)
                pred_local = torch.cat((pred_local, output_local.data), 0)

        logging.info("第{}个epoch的学习率: {}".format(epoch, optimizer_local.param_groups[0]['lr']))
        epoch_loss_val = float(running_loss_local) / float(i)
        epoch_losses_local_val.append(epoch_loss_val)
        logging.info("Local_Train_losses: {}".format(epoch_losses_local_val))

        if args.lr_scheduler == "StepLR":
            lr_scheduler_local.step()
        elif args.lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler_local.step(epoch_loss_val)
        elif args.lr_scheduler == "CosineAnnealingLR":
            lr_scheduler_local.step()

        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            save_path = os.path.join(args.output_dir, 'best_model_path/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            logging.info('Epoch [' + str(epoch) + '] [save] loss= ' + str(epoch_loss_val))
            torch.save(Local_Branch_model.state_dict(), save_path+'Best_Local'+'.pkl')
            logging.info('Best_Local_Branch_model already save!')
        else:
            logging.info('Epoch [' + str(epoch) + '] [----] loss= ' + str(epoch_loss_val))

        # log training and validation loss over each epoch
        log_train = os.path.join(args.output_dir, 'log_loss')
        with open(log_train, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if (epoch == 1):
                logwriter.writerow(["epoch", "train_loss", "val_loss", "Seed", "lr"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val, optimizer_local.param_groups[0]['lr']])

        AUROCs_l = compute_AUCs(gt, pred_local)
        AUROC_avg = np.array(AUROCs_l).mean()
        logging.info('\n')
        logging.info('Local branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(args.num_class):
            logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_l[i]))

        # save
        if AUROC_avg > AUROC_max:
            AUROC_max = AUROC_avg
            max_epoch = epoch
            save_path = os.path.join(args.output_dir, 'model_path/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(Local_Branch_model.state_dict(), save_path+'Local'+'_epoch_'+str(max_epoch)+'.pkl')
            logging.info('AUROC_max already save!')
            logging.info('Epoch [' + str(max_epoch) + '] [save] AUROC_max= ' + str(AUROC_max))

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 5):
            if epoch_loss_val > best_loss:
                logging.info("not seeing improvement in val loss")
                if ((epoch - best_epoch) >= 10):
                    logging.info("no improvement in 10 epochs")
                    # break

        logging.info("epoch:{}, AUROC_max:{}".format(max_epoch, AUROC_max))
        logging.info("best_epoch:{}, best_loss:{}".format(best_epoch, best_loss))
        time_elapsed = time.time() - since
        logging.info('Training one epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
    
if __name__ == '__main__':
    train()