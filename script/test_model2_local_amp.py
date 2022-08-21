# encoding: utf-8
import re
import sys
import os
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score, auc, roc_curve
from skimage.measure import label
from PIL import Image
from PIL import ImageFile

from dataset.build_data import ChestXrayDataSet
from model.build_local_model import build_model
from model.UNet import UNet
from utils.net_utils import Attention_gen_patchs, compute_AUCs
from main import arg_parse
args = arg_parse()
ImageFile.LOAD_TRUNCATED_IMAGES = True


# np.set_printoptions(threshold = np.nan)
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
def test(args):

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
    Local_Branch_model = build_model(args)

    model_unet = UNet(n_channels=3, n_classes=1).cuda()     # initialize model
    checkpoint = torch.load(args.UNET_PATH)
    model_unet.load_state_dict(checkpoint)      # strict=False
    logging.info("=> loaded well-trained unet model checkpoint: {}".format(args.UNET_PATH))
    model_unet.eval()

    checkpoint = torch.load(args.CKPT_PATH_L)
    Local_Branch_model.load_state_dict(checkpoint)
    logging.info("=> loaded Local_Branch_model checkpoint {}".format(args.CKPT_PATH_L))
    cudnn.benchmark = True
    logging.info('******************** load model succeed!********************')

    logging.info('******* begin testing!*********')
    test_epoch(Local_Branch_model, model_unet, test_loader)

def test_epoch(model_local, model_unet, test_loader):

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred_local = torch.FloatTensor().cuda()

    # switch to evaluate mode
    model_local.eval()
    cudnn.benchmark = True
    test_loader = tqdm(test_loader, file=sys.stdout, ncols=60)
    for i, (inp, target) in enumerate(test_loader):
        with torch.no_grad():
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            input_var = torch.autograd.Variable(inp.cuda())
            with autocast():
                var_mask = model_unet(input_var)
                output_local, _, pool_local = model_local(input_var, var_mask)
            pred_local = torch.cat((pred_local, output_local.data), 0)

    AUROCs_l = compute_AUCs(gt, pred_local)
    AUROC_avg = np.array(AUROCs_l).mean()
    logging.info('\n')
    logging.info('Local branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(args.num_class):
        logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_l[i]))

    # 绘制ROC曲线图
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    gt_np = gt.cpu().numpy()
    pred_np = pred_local.cpu().numpy()
    for i in range(args.num_class):
        fpr[i], tpr[i], _ = roc_curve(gt_np[:, i].astype(int), pred_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkblue", "darkcyan", "darkgray", "darkgreen",
                    "darkmagenta", "darkred", "lightskyblue", "yellow", "orange", "black", "red"])
    for i, color, label in zip(range(args.num_class), colors, CLASS_NAMES):
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
    plt.title("{}".format(args.model))
    plt.legend(loc="lower right")
    plt.savefig(args.output_dir + 'roc_' + '{}.png'.format(args.model))

if __name__ == '__main__':
    test()