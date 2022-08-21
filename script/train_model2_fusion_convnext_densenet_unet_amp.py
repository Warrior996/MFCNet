# encoding: utf-8
import re
import sys
import os
import csv
import time
import numpy as np
import logging
from tqdm import tqdm

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
from model.build_vit_seg import ViT_Seg
from model.UNet import UNet
from utils.net_utils import Attention_gen_patchs, compute_AUCs
from utils.build_optimizer_lr_scheduler import optimizer_lr_scheduler
from main import arg_parse
args = arg_parse()
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def train(args):

    if args.model == 'DenseNet_121':
        from model.build_model2_densenet_amp import Global_Branch, Local_Branch, Fusion_Branch
    elif args.model == 'Convnext_base':
        from model.build_model2_convnext_densenet_unet import Global_Branch, Local_Branch, Fusion_Branch

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
    Global_Branch_model = Global_Branch(pretrained=args.pretrained, num_classes=N_CLASSES).cuda()
    Local_Branch_model = Local_Branch(pretrained=args.pretrained, num_classes=N_CLASSES).cuda() 
    Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES).cuda()

    model_unet = UNet(n_channels=3, n_classes=1).cuda()  # initialize model
    checkpoint = torch.load(args.UNET_PATH)
    model_unet.load_state_dict(checkpoint)  # strict=False
    logging.info("=> loaded well-trained unet model checkpoint: {}".format(args.UNET_PATH))
    model_unet.eval()

    logging.info("Global_Branch_model:{}".format(Global_Branch_model))
    logging.info("Local_Branch_model:{}".format(Local_Branch_model))
    logging.info("Fusion_Branch_model:{}".format(Fusion_Branch_model))

    if os.path.isfile(args.CKPT_PATH_G):
        checkpoint = torch.load(args.CKPT_PATH_G)
        Global_Branch_model.load_state_dict(checkpoint)
        logging.info("[Info]: Loaded Global_Branch_model checkpoint: {}".format(args.CKPT_PATH_G))

    if os.path.isfile(args.CKPT_PATH_L):
        checkpoint = torch.load(args.CKPT_PATH_L)
        Local_Branch_model.load_state_dict(checkpoint)
        logging.info("[Info]: Loaded Local_Branch_model checkpoint: {}".format(args.CKPT_PATH_L))

    torch.backends.cudnn.benchmark = True
    criterion = nn.BCEWithLogitsLoss()
    optimizer_global, lr_scheduler_global = optimizer_lr_scheduler(Global_Branch_model, args.global_lr)
    optimizer_local, lr_scheduler_local = optimizer_lr_scheduler(Local_Branch_model, args.local_lr)
    optimizer_fusion, lr_scheduler_fusion = optimizer_lr_scheduler(Fusion_Branch_model, args.fusion_lr)

    logging.info("[Info]: Model has been loaded ...")

    logging.info("[Info]: Starting training ...")
    epoch_losses_global_train = []
    epoch_losses_local_train = []
    epoch_losses_fusion_train = []

    epoch_losses_global_val = []
    epoch_losses_local_val = []
    epoch_losses_fusion_val = []
    best_loss = 999999
    AUROC_max = 0
    scaler = GradScaler()
    for epoch in range(args.num_epochs):
        since = time.time()
        logging.info('Epoch: {}/{}'.format(epoch, args.num_epochs - 1))
        logging.info('-' * 10)
        # set the mode of model
        Global_Branch_model.eval()  # set model to training mode
        Local_Branch_model.eval()
        Fusion_Branch_model.train()

        running_loss_global = 0.0
        running_loss_local = 0.0
        running_loss_fusion = 0.0

        # Iterate over data
        train_loader = tqdm(train_loader, file=sys.stdout, ncols=60)
        for i, (input, target) in enumerate(train_loader):
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            optimizer_fusion.zero_grad(set_to_none=True)
            # compute output
            with autocast():
                output_global, fm_global, pool_global = Global_Branch_model(input_var)
                var_mask = model_unet(input_var)
                output_local, _, pool_local = Local_Branch_model(input_var, var_mask)
                output_fusion = Fusion_Branch_model(pool_global, pool_local)
                # loss
                loss_global = criterion(output_global, target_var)
                loss_local = criterion(output_local, target_var)
                loss_fusion = criterion(output_fusion, target_var)
            torch.cuda.empty_cache()
            scaler.scale(loss_fusion).backward(retain_graph=True)
            scaler.unscale_(optimizer_fusion)
            scaler.step(optimizer_fusion)
            scaler.update()

            running_loss_global += loss_global.data.item()
            running_loss_local += loss_local.data.item()
            running_loss_fusion += loss_fusion.data.item()

        epoch_loss_train = float(running_loss_global) / float(i)
        epoch_losses_global_train.append(epoch_loss_train)
        logging.info("Global_Train_losses: {}".format(epoch_losses_global_train))

        epoch_loss_train = float(running_loss_local) / float(i)
        epoch_losses_local_train.append(epoch_loss_train)
        logging.info("Local_Train_losses: {}".format(epoch_losses_local_train))

        epoch_loss_train = float(running_loss_fusion) / float(i)
        epoch_losses_fusion_train.append(epoch_loss_train)
        logging.info("Fusion_Train_losses: {}".format(epoch_losses_fusion_train))


        logging.info("[Info]: Starting valing ...")
        gt = torch.FloatTensor().cuda()
        pred_global = torch.FloatTensor().cuda()
        pred_local = torch.FloatTensor().cuda()
        pred_fusion = torch.FloatTensor().cuda()
        # switch to evaluate mode
        Global_Branch_model.eval()
        Local_Branch_model.eval()
        Fusion_Branch_model.eval()
        cudnn.benchmark = True
    
        running_loss = 0.0
        running_loss_global = 0.0
        running_loss_local = 0.0
        running_loss_fusion = 0.0
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=60)
        for i, (inp, target) in enumerate(val_loader):
            with torch.no_grad():
                target = target.cuda()
                gt = torch.cat((gt, target), 0)
                input_var = torch.autograd.Variable(inp.cuda())
                with autocast():
                    output_global, fm_global, pool_global = Global_Branch_model(input_var)
                    var_mask = model_unet(input_var)
                    output_local, _, pool_local = Local_Branch_model(input_var, var_mask)
                    output_fusion = Fusion_Branch_model(pool_global, pool_local)

                    loss_global = criterion(output_global, target)
                    loss_local = criterion(output_local, target)
                    loss_fusion = criterion(output_fusion, target)

                running_loss_global += loss_global.data.item()
                running_loss_local += loss_local.data.item()
                running_loss_fusion += loss_fusion.data.item()

                pred_global = torch.cat((pred_global, output_global.data), 0)
                pred_local = torch.cat((pred_local, output_local.data), 0)
                pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)
        
        logging.info("第{}个epoch的学习率: {}".format(epoch, optimizer_global.param_groups[0]['lr']))
        logging.info("第{}个epoch的学习率: {}".format(epoch, optimizer_local.param_groups[0]['lr']))
        logging.info("第{}个epoch的学习率: {}".format(epoch, optimizer_fusion.param_groups[0]['lr']))

        epoch_loss_val = float(running_loss_global) / float(i)
        epoch_losses_global_val.append(epoch_loss_val)
        logging.info("Global_Train_losses: {}".format(epoch_losses_global_val))

        epoch_loss_val = float(running_loss_local) / float(i)
        epoch_losses_local_val.append(epoch_loss_val)
        logging.info("Local_Train_losses: {}".format(epoch_losses_local_val))

        epoch_loss_val = float(running_loss_fusion) / float(i)
        epoch_losses_fusion_val.append(epoch_loss_val)
        logging.info("Fusion_Train_losses: {}".format(epoch_losses_fusion_val))

        if args.lr_scheduler == "StepLR":
            lr_scheduler_fusion.step()
        elif args.lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler_fusion.step(epoch_loss_val)

        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            save_path = os.path.join(args.output_dir, 'best_model_path/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            logging.info('Epoch [' + str(epoch) + '] [save] loss= ' + str(epoch_loss_val))
            torch.save(Fusion_Branch_model.state_dict(), save_path+'Best_Fusion'+'.pkl')            
            logging.info('Best_Fusion_Branch_model already save!')
        else:
            logging.info('Epoch [' + str(epoch) + '] [----] loss= ' + str(epoch_loss_val))
        # log training and validation loss over each epoch
        log_train = os.path.join(args.output_dir, 'log_loss')
        with open(log_train, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if (epoch == 1):
                logwriter.writerow(["epoch", "train_loss", "val_loss", "Seed", "lr"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val, optimizer_fusion.param_groups[0]['lr']])

        AUROCs_g = compute_AUCs(gt, pred_global)
        AUROC_avg = np.array(AUROCs_g).mean()
        logging.info('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_g[i]))

        AUROCs_l = compute_AUCs(gt, pred_local)
        AUROC_avg = np.array(AUROCs_l).mean()
        logging.info('\n')
        logging.info('Local branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_l[i]))

        AUROCs_f = compute_AUCs(gt, pred_fusion)
        AUROC_avg = np.array(AUROCs_f).mean()
        logging.info('\n')
        logging.info('Fusion branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_f[i]))
            
        # save
        if AUROC_avg > AUROC_max:
            AUROC_max = AUROC_avg
            max_epoch = epoch
            save_path = os.path.join(args.output_dir, 'model_path/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(Fusion_Branch_model.state_dict(), save_path+'Fusion'+'_epoch_'+str(epoch)+'.pkl')
            logging.info('AUROC_max already save!')
            logging.info('Epoch [' + str(max_epoch) + '] [save] AUROC_max= ' + str(AUROC_max))

          # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 5):
            if epoch_loss_val > best_loss:
                logging.info("not seeing improvement in val loss")
                if ((epoch - best_epoch) >= 10):
                    logging.info("no improvement in 10 epochs")
                    # break
        if AUROC_avg > AUROC_max:
            AUROC_max = AUROC_avg
            max_epoch = epoch
        logging.info("epoch:{}, AUROC_max:{}".format(max_epoch, AUROC_max))
        logging.info("best_epoch:{}, best_loss:{}".format(best_epoch, best_loss))
        time_elapsed = time.time() - since
        logging.info('Training one epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
    
if __name__ == '__main__':
    train()