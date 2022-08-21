# encoding: utf-8
import logging
import re
import sys
import os
from tabnanny import verbose
import cv2
import csv
import time
from tqdm import tqdm
import numpy as np
import argparse
import logging
from utils.logging import open_log
import matplotlib.pyplot as plt
import torch
from torch._C import device
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset.build_data import ChestXrayDataSet
from model.build_global_model import DenseNet121, Convnext_base
from utils.net_utils import Attention_gen_patchs, compute_AUCs

# np.set_printoptions(threshold = np.nan)

N_CLASSES = 14
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
    Global_Branch_model = DenseNet121(pretrained = args.pretrained, num_classes = N_CLASSES).cuda()
    # 是否冻结权重
    if args.freeze_layers:
        for name, para in Global_Branch_model.named_parameters():
            if "classifier" not in name:
                para.requires_grad_(False)
        logging.info("[Info]: 除最后的全连接层外，其他权重全部冻结")
    else:
        logging.info("[Info]: 没有权重全部冻结")
    # logging.info('model:{}'.format(Global_Branch_model))

    cudnn.benchmark = True
    pg = [p for p in Global_Branch_model.parameters() if p.requires_grad]
    criterion = nn.BCEWithLogitsLoss()
    if args.optim == "Adam":
        optimizer_global = optim.Adam(pg, lr=args.global_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    elif args.optim == "SGD":
        optimizer_global = optim.SGD(pg, lr=args.global_lr, momentum=0.9, weight_decay=0.0001)

    scaler = GradScaler()
    if args.lr_scheduler == "StepLR":
        lr_scheduler_global = lr_scheduler.StepLR(optimizer_global, step_size = 30, gamma = 0.1, verbose=True)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler_global = lr_scheduler.ReduceLROnPlateau(optimizer_global, mode='min', factor=0.1, patience=5, verbose=True)
    
    # lr_scheduler_global = lr_scheduler.CosineAnnealingLR(optimizer_global, T_max=20, verbose=True)
    
    logging.info("[Info]: Model has been loaded ...")

    logging.info("[Info]: Starting training ...")
    epoch_losses_train = []
    epoch_losses_val = []
    best_loss = 999999

    for epoch in range(args.num_epochs):
        since = time.time()
        logging.info('Epoch: {}/{}'.format(epoch, args.num_epochs - 1))
        logging.info('-' * 10)
        #set the mode of model
        Global_Branch_model.train()  # set model to training mode
        running_loss = 0.0

        # Iterate over data
        train_loader = tqdm(train_loader, file=sys.stdout, ncols=60)

        for i, (input, target) in enumerate(train_loader):
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            optimizer_global.zero_grad()
            output_global = Global_Branch_model(input_var)
            loss = criterion(output_global, target_var)
            torch.cuda.empty_cache()
            # loss.backward() 
            loss.backward() 
            optimizer_global.step()  
            running_loss += loss.data.item()

     
        epoch_loss_train = float(running_loss) / float(i)
        epoch_losses_train.append(epoch_loss_train)
        logging.info("Train_losses: {}".format(epoch_losses_train))

        logging.info("[Info]: Starting valing ...")
        gt = torch.FloatTensor().cuda()
        pred_global = torch.FloatTensor().cuda()
        # switch to evaluate mode
        Global_Branch_model.eval()

        cudnn.benchmark = True
    
        running_loss = 0.0
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=60)
        for i, (inp, target) in enumerate(val_loader):
            with torch.no_grad():
                target = target.cuda()
                gt = torch.cat((gt, target), 0)
                input_var = torch.autograd.Variable(inp.cuda())
                output_global = Global_Branch_model(input_var)
                loss = criterion(output_global, target)
                running_loss += loss.data.item()
                pred_global = torch.cat((pred_global, output_global.data), 0)
      
        logging.info("第{}个epoch的学习率: {}".format(epoch, optimizer_global.param_groups[0]['lr']))
      
        epoch_loss_val = float(running_loss) / float(i)
        epoch_losses_val.append(epoch_loss_val)
        logging.info("val_losses: {}".format(epoch_losses_val))
       
        if args.lr_scheduler == "StepLR":
            lr_scheduler_global.step()
        elif args.lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler_global.step(epoch_loss_val)
       
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            save_path = os.path.join(args.output_dir, 'best_model_path/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            logging.info ('Epoch [' + str(epoch) + '] [save] loss= ' + str(epoch_loss_val))
            torch.save(Global_Branch_model.state_dict(), save_path+'Best_Global'+'_epoch_'+str(epoch)+'.pkl')
            torch.save(Global_Branch_model.state_dict(), save_path+'Best_Global'+'.pkl')
            logging.info('Best_Global_Branch_model already save!')

        else:
            logging.info ('Epoch [' + str(epoch) + '] [----] loss= ' + str(epoch_loss_val))

        # log training and validation loss over each epoch
        log_train = os.path.join(args.output_dir, 'log_train')
        with open(log_train, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if (epoch == 1):
                logwriter.writerow(["epoch", "train_loss", "val_loss","Seed"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val])

        AUROCs_g = compute_AUCs(gt, pred_global)
        AUROC_avg = np.array(AUROCs_g).mean()
        logging.info('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_g[i]))

        #save
        if epoch % 1 == 0:
            save_path = os.path.join(args.output_dir, 'model_path/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(Global_Branch_model.state_dict(), save_path+'Global'+'_epoch_'+str(epoch)+'.pkl')
            logging.info('Global_Branch_model already save!')

          # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            if epoch_loss_val > best_loss:
                logging.info("not seeing improvement in val loss")
                if ((epoch - best_epoch) >= 10):
                    logging.info("no improvement in 10 epochs")
                    # break
        logging.info("best_epoch:{}, best_loss:{}".format(best_epoch, best_loss))
        time_elapsed = time.time() - since
        logging.info('Training one epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))

if __name__ == '__main__':
    train()