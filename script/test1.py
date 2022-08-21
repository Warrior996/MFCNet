# encoding: utf-8
import re
import sys
import os
import cv2
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from skimage.measure import label
from PIL import Image
from PIL import ImageFile

from dataset.build_data import ChestXrayDataSet
from model.build_global_model import Densenet121_AG, Fusion_Branch, DenseNet121, ImageClassifier
# from model.build_model2_convnext import Global_Branch, Local_Branch, Fusion_Branch
from model.build_model2_densenet import Global_Branch, Local_Branch, Fusion_Branch
from model.UNet import UNet
from utils.net_utils import Attention_gen_patchs, compute_AUCs, make_print_to_file
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.logging import open_log

# np.set_printoptions(threshold = np.nan)


CKPT_PATH = ''

CKPT_PATH_G = 'results/best_model_path/Best_Global.pkl'
CKPT_PATH_L = 'results/best_model_path/Best_Local.pkl'
CKPT_PATH_F = 'results/best_model_path/Best_Fusion.pkl'
UNET_PATH = '../previous_models/best_unet.pkl'

N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

DATA_DIR = '../../../data/ChestX-ray14/images'
TRAIN_IMAGE_LIST = '/labels/train_list.txt'
TEST_IMAGE_LIST = './labels/test_list.txt'

BATCH_SIZE = 64

def test(args):

    if args.model == 'densenet':
        from model.build_model2_densenet import Global_Branch, Local_Branch, Fusion_Branch
    elif args.model == 'convnext':
        from model.build_model2_convnext import Global_Branch, Local_Branch, Fusion_Branch  
    # make_print_to_file(path='./results/log_test/')
    # 这里输出之后的所有的输出的print 内容即将写入日志
    logging.info('********************load data********************')
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=args.dataset_dir,
                                    image_list_file=args.testSet,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    logging.info('********************load data succeed!********************')


    logging.info('********************load model********************')
    # initialize and load the model
    Global_Branch_model = DenseNet121(pretrained = args.pretrained, num_classes = N_CLASSES).cuda()
    # Global_Branch_model = Global_Branch(pretrained = args.pretrained, num_classes = N_CLASSES).cuda()
    Local_Branch_model = Local_Branch(pretrained = args.pretrained, num_classes = N_CLASSES).cuda() 
    model_unet = UNet(n_channels=3, n_classes=1).cuda()     #initialize model 
    if os.path.exists(UNET_PATH):
        checkpoint = torch.load(UNET_PATH)
        model_unet.load_state_dict(checkpoint)      #strict=False
        logging.info("=> loaded well-trained unet model checkpoint: {}".format(UNET_PATH))
    model_unet.eval()
    Fusion_Branch_model = Fusion_Branch(input_size = 2048, output_size = N_CLASSES).cuda()
   
    if os.path.isfile(CKPT_PATH_G):
        checkpoint = torch.load(CKPT_PATH_G)
        Global_Branch_model.load_state_dict(checkpoint)
        logging.info("=> loaded Global_Branch_model checkpoint {}".format(CKPT_PATH_G))

    if os.path.isfile(CKPT_PATH_L):
        checkpoint = torch.load(CKPT_PATH_L)
        Local_Branch_model.load_state_dict(checkpoint)
        logging.info("=> loaded Local_Branch_model checkpoint {}".format(CKPT_PATH_L))

    if os.path.isfile(CKPT_PATH_F):
        checkpoint = torch.load(CKPT_PATH_F)
        Fusion_Branch_model.load_state_dict(checkpoint)
        logging.info("=> loaded Fusion_Branch_model checkpoint {}".format(CKPT_PATH_F))

    cudnn.benchmark = True
    logging.info('******************** load model succeed!********************')

    logging.info('******* begin testing!*********')
    test_epoch(Global_Branch_model, Local_Branch_model, Fusion_Branch_model, model_unet, test_loader)

def test_epoch(model_global, model_local, model_fusion, model_unet, test_loader):

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred_global = torch.FloatTensor().cuda()
    pred_local = torch.FloatTensor().cuda()
    pred_fusion = torch.FloatTensor().cuda()

    # switch to evaluate mode
    model_global.eval()
    model_local.eval()
    model_fusion.eval()
    cudnn.benchmark = True

    for i, (inp, target) in enumerate(test_loader):
        with torch.no_grad():
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            input_var = torch.autograd.Variable(inp.cuda())
            output_global = model_global(input_var)
            
            # output_global, fm_global, pool_global = model_global(input_var)
            
            # var_mask = model_unet(input_var)
            # output_local, _, pool_local = model_local(input_var, var_mask) 
            # patchs_var = Attention_gen_patchs(inp,fm_global)
            # patchs_var = torch.autograd.Variable(patchs_var.cuda())
            # output_local, _, pool_local = model_local(patchs_var)

            # output_fusion = model_fusion(pool_global,pool_local)

            pred_global = torch.cat((pred_global, output_global.data), 0)
            # pred_local = torch.cat((pred_local, output_local.data), 0)
            # pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)
            if (i % 200 == 0):
                print(str(i * BATCH_SIZE))
            
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


if __name__ == '__main__':
    test()