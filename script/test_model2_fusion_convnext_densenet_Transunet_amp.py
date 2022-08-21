# encoding: utf-8
import re
import sys
import os
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from PIL import ImageFile


import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset.build_data import ChestXrayDataSet
from model.build_vit_seg import ViT_Seg
from utils.net_utils import Attention_gen_patchs, compute_AUCs
from main import arg_parse
args = arg_parse()

ImageFile.LOAD_TRUNCATED_IMAGES = True
# np.set_printoptions(threshold = np.nan)

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def test(args):

    if args.model == 'DenseNet_121':
        from model.build_model2_densenet_amp import Global_Branch, Local_Branch, Fusion_Branch
    elif args.model == 'Convnext_base':
        from model.build_model2_convnext_densenet import Global_Branch, Local_Branch, Fusion_Branch
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
                             shuffle=False, num_workers=args.num_works, pin_memory=True)
    logging.info('********************load data succeed!********************')

    logging.info('********************load model********************')
    # initialize and load the model
    Global_Branch_model = Global_Branch(pretrained=args.pretrained, num_classes=N_CLASSES).cuda()
    Local_Branch_model = Local_Branch(pretrained=args.pretrained, num_classes=N_CLASSES).cuda()
    Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES).cuda()
    model_vit_seg = ViT_Seg()

    if os.path.isfile(args.CKPT_PATH_G):
        checkpoint = torch.load(args.CKPT_PATH_G)
        Global_Branch_model.load_state_dict(checkpoint)
        logging.info("=> loaded Global_Branch_model checkpoint {}".format(args.CKPT_PATH_G))

    if os.path.isfile(args.CKPT_PATH_L):
        checkpoint = torch.load(args.CKPT_PATH_L)
        Local_Branch_model.load_state_dict(checkpoint)
        logging.info("=> loaded Local_Branch_model checkpoint {}".format(args.CKPT_PATH_L))

    if os.path.isfile(args.CKPT_PATH_F):
        checkpoint = torch.load(args.CKPT_PATH_F)
        Fusion_Branch_model.load_state_dict(checkpoint)
        logging.info("=> loaded Fusion_Branch_model checkpoint {}".format(args.CKPT_PATH_F))

    cudnn.benchmark = True
    logging.info('******************** load model succeed!********************')

    logging.info('******* begin testing!*********')
    test_epoch(Global_Branch_model, Local_Branch_model, Fusion_Branch_model, model_vit_seg, test_loader)

def test_epoch(model_global, model_local, model_fusion, model_vit_seg, test_loader):

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
    test_loader = tqdm(test_loader, file=sys.stdout, ncols=60)
    for i, (inp, target) in enumerate(test_loader):
        with torch.no_grad():
            target = target.cuda()
            input_var = inp.cuda()
            with autocast():
                output_global, fm_global, pool_global = model_global(input_var)
                var_mask = model_vit_seg(input_var)
                output_local, _, pool_local = model_local(input_var, var_mask)
                output_fusion = model_fusion(pool_global, pool_local)

            gt = torch.cat((gt, target), 0)
            pred_global = torch.cat((pred_global, output_global.data), 0)
            pred_local = torch.cat((pred_local, output_local.data), 0)
            pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)

    AUROCs_g = compute_AUCs(gt, pred_global)
    AUROC_avg = np.array(AUROCs_g).mean()
    logging.info('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_g[i]))

    # 绘制ROC曲线图
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    gt_np = gt.cpu().numpy()
    pred_np = pred_global.cpu().numpy()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(gt_np[:, i].astype(int), pred_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkblue", "darkcyan", "darkgray", "darkgreen",
                    "darkmagenta", "darkred", "lightskyblue", "yellow", "orange", "black", "red"])
    for i, color, label in zip(range(N_CLASSES), colors, CLASS_NAMES):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="{0} (area = {1:0.2f})".format(label, roc_auc[i]),
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Global_"+"{}".format(args.model))
    plt.legend(loc="lower right")
    plt.savefig(args.output_dir + 'Global_roc_' + '{}_{}.png'.format(args.branch, args.model))

    AUROCs_l = compute_AUCs(gt, pred_local)
    AUROC_avg = np.array(AUROCs_l).mean()
    logging.info('\n')
    logging.info('Local branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_l[i]))

    # 绘制ROC曲线图
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    gt_np = gt.cpu().numpy()
    pred_np = pred_local.cpu().numpy()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(gt_np[:, i].astype(int), pred_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkblue", "darkcyan", "darkgray", "darkgreen",
                    "darkmagenta", "darkred", "lightskyblue", "yellow", "orange", "black", "red"])
    for i, color, label in zip(range(N_CLASSES), colors, CLASS_NAMES):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="{0} (area = {1:0.2f})".format(label, roc_auc[i]),
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Local_"+"{}".format(args.model))
    plt.legend(loc="lower right")
    plt.savefig(args.output_dir + 'Local_roc_' + '{}_{}.png'.format(args.branch, args.model))

    AUROCs_f = compute_AUCs(gt, pred_fusion)
    AUROC_avg = np.array(AUROCs_f).mean()
    logging.info('\n')
    logging.info('Fusion branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_f[i]))

    # 绘制ROC曲线图
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    gt_np = gt.cpu().numpy()
    pred_np = pred_fusion.cpu().numpy()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(gt_np[:, i].astype(int), pred_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkblue", "darkcyan", "darkgray", "darkgreen",
                    "darkmagenta", "darkred", "lightskyblue", "yellow", "orange", "black", "red"])
    for i, color, label in zip(range(N_CLASSES), colors, CLASS_NAMES):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="{0} (area = {1:0.2f})".format(label, roc_auc[i]),
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fusion_"+"{}".format(args.model))
    plt.legend(loc="lower right")
    plt.savefig(args.output_dir + 'Fusion_roc_' + '{}_{}.png'.format(args.branch, args.model))

if __name__ == '__main__':
    test()