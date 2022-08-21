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
from sklearn.metrics import roc_curve, auc, roc_auc_score
from skimage.measure import label
from PIL import Image
from PIL import ImageFile

from dataset.build_data import ChestXrayDataSet
from model.build_global_model import build_model
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
    Global_Branch_model = build_model()

    checkpoint = torch.load(args.CKPT_PATH_G)
    Global_Branch_model.load_state_dict(checkpoint)
    logging.info("=> loaded Global_Branch_model checkpoint {}".format(args.CKPT_PATH_G))
    cudnn.benchmark = True
    logging.info('******************** load model succeed!********************')

    logging.info('******* begin testing!*********')
    test_epoch(Global_Branch_model, test_loader)

def test_epoch(model_global, test_loader):

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred_global = torch.FloatTensor().cuda()

    # switch to evaluate mode
    model_global.eval()
    cudnn.benchmark = True
    test_loader = tqdm(test_loader, file=sys.stdout, ncols=60)
    for i, (inp, target) in enumerate(test_loader):
        with torch.no_grad():
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            input_var = torch.autograd.Variable(inp.cuda())
            with autocast():
                output_local = model_global(input_var)
            pred_global = torch.cat((pred_global, output_local.data), 0)

    AUROCs_g = compute_AUCs(gt, pred_global)
    AUROC_avg = np.array(AUROCs_g).mean()
    logging.info('\n')
    logging.info('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(args.num_class):
        logging.info('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_g[i]))

    # 绘制ROC曲线图
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    gt_np = gt.cpu().numpy()
    pred_np = pred_global.cpu().numpy()
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