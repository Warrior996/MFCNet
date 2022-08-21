# encoding: utf-8
import logging
import re
import sys
import os
from tabnanny import verbose
import csv
import time
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
from dataset.build_data import ChestXrayDataSet
from model.build_global_model import Densenet121_AG, Fusion_Branch, DenseNet121, Convnext_base, Pyconv_densenet121
from utils.net_utils import Attention_gen_patchs, compute_AUCs, make_print_to_file

# np.set_printoptions(threshold = np.nan)

CKPT_PATH = ''

CKPT_PATH_G = 'previous_models/AG_CNN_Global_epoch_1.pkl' 
CKPT_PATH_L = 'previous_models/AG_CNN_Local_epoch_2.pkl' 
CKPT_PATH_F = 'previous_models/AG_CNN_Fusion_epoch_2.pkl'

CKPT_PATH_G = '' 
CKPT_PATH_L = '' 
CKPT_PATH_F = ''

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
    Global_Branch_model = Densenet121_AG(pretrained = False, num_classes = config['num_classes']).cuda()
    Local_Branch_model = Densenet121_AG(pretrained = False, num_classes = config['num_classes']).cuda()
    if args.model == 'densenet':
        Global_Branch_model = DenseNet121(pretrained = args.pretrained, num_classes = N_CLASSES).cuda()

    # Global_Branch_model = Pyconv_densenet121(pretrained = args.pretrained, num_classes = N_CLASSES).cuda()
    # Local_Branch_model = DenseNet121(pretrained = True, num_classes = N_CLASSES).cuda()
    elif args.model == 'convnext':
        Global_Branch_model = Convnext_base(pretrained = args.pretrained, num_classes = N_CLASSES).cuda()
    # Local_Branch_model = Convnext_base(pretrained = True, num_classes = config['num_classes']).cuda()
    # Fusion_Branch_model = Fusion_Branch(input_size = 2048, output_size = N_CLASSES).cuda()
    logging.info('model:{}'.format(Global_Branch_model))
    # if os.path.isfile(CKPT_PATH):
    #     logging.info("[Info]: Loading checkpoint")
    #     checkpoint = torch.load(CKPT_PATH)
    #     # to load state
    #     # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
    #     state_dict = checkpoint['state_dict']
    #     remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)

    #     pattern = re.compile(
    #         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    #     for key in list(state_dict.keys()):
    #         ori_key =  key
    #         key = key.replace('densenet121.','')
    #         #print('key',key)
    #         match = pattern.match(key)
    #         new_key = match.group(1) + match.group(2) if match else key
    #         new_key = new_key[7:] if remove_data_parallel else new_key
    #         #print('new_key',new_key)
    #         if '.0.' in new_key:
    #             new_key = new_key.replace('0.','')
    #         state_dict[new_key] = state_dict[ori_key]
    #         # Delete old key only if modified.
    #         if match or remove_data_parallel: 
    #             del state_dict[ori_key]
        
    #     Global_Branch_model.load_state_dict(state_dict)
    #     Local_Branch_model.load_state_dict(state_dict)
    #     logging.info("[Info]: Loaded baseline checkpoint")
        
    # else:
    #     logging.info("[Info]: No previous checkpoint found ...")

    # if os.path.isfile(CKPT_PATH_G):
    #     checkpoint = torch.load(CKPT_PATH_G)
    #     Global_Branch_model.load_state_dict(checkpoint)
    #     logging.info("[Info]: Loaded Global_Branch_model checkpoint")

    # if os.path.isfile(CKPT_PATH_L):
    #     checkpoint = torch.load(CKPT_PATH_L)
    #     Local_Branch_model.load_state_dict(checkpoint)
    #     logging.info("[Info]: Loaded Local_Branch_model checkpoint")

    # if os.path.isfile(CKPT_PATH_F):
    #     checkpoint = torch.load(CKPT_PATH_F)
    #     Fusion_Branch_model.load_state_dict(checkpoint)
    #     logging.info("[Info]: Loaded Fusion_Branch_model checkpoint")

    cudnn.benchmark = True
    criterion = nn.BCELoss()
    if args.optim == "Adam":
        optimizer_global = optim.Adam(Global_Branch_model.parameters(), lr=args.global_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    elif args.optim == "SGD":
        optimizer_global = optim.SGD(Global_Branch_model.parameters(), lr=args.global_lr, momentum=0.9, weight_decay=0.0001)

    # lr_scheduler_global = lr_scheduler.ReduceLROnPlateau(optimizer_global, mode='min', factor=0.1, patience=5, verbose=True)
    lr_scheduler_global = lr_scheduler.StepLR(optimizer_global, step_size=30, gamma=0.1, verbose=True)
    # lr_scheduler_global = lr_scheduler.CosineAnnealingLR(optimizer_global, T_max=20, verbose=True)
    
    # optimizer_local = optim.Adam(Local_Branch_model.parameters(), lr=LR_L, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # lr_scheduler_local = lr_scheduler.StepLR(optimizer_local, step_size = 10, gamma = 1, verbose=True)
    
    # optimizer_fusion = optim.Adam(Fusion_Branch_model.parameters(), lr=LR_F, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # lr_scheduler_fusion = lr_scheduler.StepLR(optimizer_fusion, step_size = 15, gamma = 1, verbose=True)
    logging.info("[Info]: Model has been loaded ...")

    logging.info("[Info]: Starting training ...")
    epoch_losses_train = []
    epoch_losses_val = []
    best_loss = 999999

    y = []
    for epoch in range(args.num_epochs):
        since = time.time()
        logging.info('Epoch: {}/{}'.format(epoch , args.num_epochs - 1))
        logging.info('-' * 10)
        #set the mode of model
        Global_Branch_model.train()  #set model to training mode
        # Local_Branch_model.train()
        # Fusion_Branch_model.train()

        running_loss = 0.0

        #Iterate over data
        for i, (input, target) in enumerate(train_loader):
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            optimizer_global.zero_grad()
            # optimizer_local.zero_grad()
            # optimizer_fusion.zero_grad()

            # compute output
            # output_global, fm_global, pool_global = Global_Branch_model(input_var)
            output_global = Global_Branch_model(input_var)
            # patchs_var = Attention_gen_patchs(input, fm_global).cuda()
            # torch.cuda.empty_cache()
            # output_local, _, pool_local = Local_Branch_model(patchs_var)
            # #print(fusion_var.shape)
            # output_fusion = Fusion_Branch_model(pool_global, pool_local)
            #
            torch.cuda.empty_cache()
            # loss
            loss = criterion(output_global, target_var)
            # loss2 = criterion(output_local, target_var)
            # loss3 = criterion(output_fusion, target_var)
            #

            # loss = loss1*0.8 + loss2*0.1 + loss3*0.1 

            loss.backward() 
            optimizer_global.step()  
            # optimizer_local.step()
            # optimizer_fusion.step()
       
            #print(loss.data.item())
            running_loss += loss.data.item()
            if (i % 200 == 0):
                print(str(i * args.batch_size))
     
        epoch_loss_train = float(running_loss) / float(i)
        epoch_losses_train.append(epoch_loss_train)
        logging.info("Train_losses: {}".format(epoch_losses_train))
        # print(' Epoch over  Loss: {:.5f}'.format(epoch_loss))

        logging.info("[Info]: Starting valing ...")
        gt = torch.FloatTensor().cuda()
        pred_global = torch.FloatTensor().cuda()
        pred_local = torch.FloatTensor().cuda()
        pred_fusion = torch.FloatTensor().cuda()
        criterion = nn.BCELoss()
        # switch to evaluate mode
        Global_Branch_model.eval()
        # Local_Branch_model.eval()
        # Fusion_Branch_model.eval()
        cudnn.benchmark = True
    
        running_loss = 0.0
    
        for i, (inp, target) in enumerate(val_loader):
            with torch.no_grad():
                target = target.cuda()
                gt = torch.cat((gt, target), 0)
                input_var = torch.autograd.Variable(inp.cuda())
                #output = model_global(input_var)

                # output_global, fm_global, pool_global = Global_Branch_model(input_var)
                output_global = Global_Branch_model(input_var)
                # patchs_var = Attention_gen_patchs(inp, fm_global).cuda()

                # output_local, _, pool_local = Local_Branch_model(patchs_var)

                # output_fusion = Fusion_Branch_model(pool_global,pool_local)

                loss = criterion(output_global, target)
                # loss2 = criterion(output_local, target)
                # loss3 = criterion(output_fusion, target)
                # loss = loss1*0.8 + loss2*0.1 + loss3*0.1 
                running_loss += loss.data.item()
                pred_global = torch.cat((pred_global, output_global.data), 0)
                # pred_local = torch.cat((pred_local, output_local.data), 0)
                # pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)
                if (i % 200 == 0):
                    print(str(i * args.batch_size))

        logging.info("第{}个epoch的学习率: {}".format(epoch, optimizer_global.param_groups[0]['lr']))
        lr_scheduler_global.step()
        # lr_scheduler_local.step() 
        # lr_scheduler_fusion.step() 
        # y.append(lr_scheduler_global.get_last_lr()[0])
        # logging.info("y: {}".format(y))
        epoch_loss_val = float(running_loss) / float(i)
        epoch_losses_val.append(epoch_loss_val)
        logging.info("val_losses: {}".format(epoch_losses_val))

        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            save_path = os.path.join(args.output_dir, 'best_model_path/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            logging.info ('Epoch [' + str(epoch) + '] [save] loss= ' + str(epoch_loss_val))
            torch.save(Global_Branch_model.state_dict(), save_path+'Best_Global'+'.pkl')
            logging.info('Best_Global_Branch_model already save!')
            # torch.save(Local_Branch_model.state_dict(), save_path+'Best_Local'+'.pkl')
            # logging.info('Best_Local_Branch_model already save!')
            # torch.save(Fusion_Branch_model.state_dict(), save_path+'Best_Fusion'+'.pkl')            
            # logging.info('Best_Fusion_Branch_model already save!')
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

        # AUROCs_l = compute_AUCs(gt, pred_local)
        # AUROC_avg = np.array(AUROCs_l).mean()
        # logging.info('\n')
        # logging.info('Local branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        # for i in range(N_CLASSES):
        #     logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_l[i]))

        # AUROCs_f = compute_AUCs(gt, pred_fusion)
        # AUROC_avg = np.array(AUROCs_f).mean()
        # logging.info('\n')
        # logging.info('Fusion branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        # for i in range(N_CLASSES):
        #     logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_f[i]))
            
        #save
        if epoch % 1 == 0:
            save_path = os.path.join(args.output_dir, 'model_path/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(Global_Branch_model.state_dict(), save_path+'Global'+'_epoch_'+str(epoch)+'.pkl')
            logging.info('Global_Branch_model already save!')
            # torch.save(Local_Branch_model.state_dict(), save_path+'Local'+'_epoch_'+str(epoch)+'.pkl')
            # logging.info('Local_Branch_model already save!')
            # torch.save(Fusion_Branch_model.state_dict(), save_path+'Fusion'+'_epoch_'+str(epoch)+'.pkl')            
            # logging.info('Fusion_Branch_model already save!')

          # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            if epoch_loss_val > best_loss:
                logging.info("not seeing improvement in val loss")
                # LR_F = LR_F / 2
                # LR_G = LR_G / 2
                # LR_L = LR_L / 2
                # logging.info("created new optimizer with LR " + str(LR_F))
                if ((epoch - best_epoch) >= 10):
                    logging.info("no improvement in 10 epochs")
                    logging.info("best epoch: {}  best loss:{}".format(best_epoch, best_loss))
                    # break
        time_elapsed = time.time() - since
        logging.info('Training one epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))

if __name__ == '__main__':
    train()