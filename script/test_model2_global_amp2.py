# encoding: utf-8
import re
import sys
import os
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
from itertools import cycle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import roc_curve, auc, roc_auc_score, \
    average_precision_score, precision_recall_curve

from dataset.build_data import ChestXrayDataSet
from dataset.NIH import NIH
from model.build_global_model import build_model
from utils.net_utils import compute_AUCs
from main import arg_parse
args = arg_parse()
ImageFile.LOAD_TRUNCATED_IMAGES = True

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('********************load data********************')
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    # test_dataset = ChestXrayDataSet(data_dir=args.dataset_dir,
    #                                 image_list_file=args.testSet,
    #                                 transform=transforms.Compose([
    #                                     transforms.Resize(256),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     normalize,
    #                                 ]))
    # test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=args.num_works, pin_memory=True)
    # val_dataset = ChestXrayDataSet(data_dir=args.dataset_dir,
    #                                image_list_file=args.valSet,
    #                                transform=transforms.Compose([
    #                                    transforms.Resize(256),
    #                                    transforms.CenterCrop(224),
    #                                    transforms.ToTensor(),
    #                                    normalize,
    #                                ]))
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
    #                         shuffle=False, num_workers=args.num_works, pin_memory=True)

    val_df = pd.read_csv("labels/val.csv")
    test_df = pd.read_csv("labels/test.csv")
    size = len(test_df)
    logging.info("Test _df size: {}".format(size))
    size = len(val_df)
    logging.info("val_df size: {}".format(size))


    dataset_test = NIH(test_df, path_image=args.dataset_dir, transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize]))
    test_loader = DataLoader(dataset_test, args.batch_size, shuffle=False,
                             num_workers=args.num_works, pin_memory=True)

    dataset_val = NIH(val_df, path_image=args.dataset_dir, transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize]))
    val_loader = DataLoader(dataset_val, args.batch_size, shuffle=True,
                            num_workers=args.num_works, pin_memory=True)

    logging.info('********************load data succeed!********************')

    logging.info('********************load model********************')
    # initialize and load the model
    Global_Branch_model= build_model()

    if os.path.isfile(args.CKPT_PATH_G):
        checkpoint = torch.load(args.CKPT_PATH_G)
        Global_Branch_model.load_state_dict(checkpoint)
        logging.info("=> loaded Global_Branch_model checkpoint {}".format(args.CKPT_PATH_G))

    cudnn.benchmark = True
    logging.info('******************** load model succeed!********************')
    for mode in ["Threshold", "test"]:
        # create empty dfs
        pred_df = pd.DataFrame(columns=["path"])
        bi_pred_df = pd.DataFrame(columns=["path"])
        true_df = pd.DataFrame(columns=["path"])
        gt = torch.FloatTensor().cuda()
        pred_global = torch.FloatTensor().cuda()


        if mode == "Threshold":
            loader = val_loader
            Eval_df = pd.DataFrame(columns=["label", 'bestthr'])
            thrs = []

        if mode == "test":
            loader = test_loader
            TestEval_df = pd.DataFrame(columns=["label", 'auc', "auprc"])

            Eval = pd.read_csv(args.output_dir + "Threshold.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"] == "Atelectasis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Cardiomegaly"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Effusion"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Infiltration"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Mass"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Nodule"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumonia"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumothorax"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Consolidation"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Edema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Emphysema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Fibrosis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pleural_Thickening"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Hernia"].index[0]]]
        loader = tqdm(loader, file=sys.stdout, ncols=60)
        for i, data in enumerate(loader):
            inputs, labels, item = data
            inputs, labels = inputs.to(device), labels.to(device)
            true_labels = labels.cpu().data.numpy()
            batch_size = true_labels.shape
            Global_Branch_model.eval()
            with torch.no_grad():
                with autocast():
                    outputs = Global_Branch_model(inputs)
                    outputs = torch.sigmoid(outputs)
            probs = outputs.cpu().data.numpy()

            gt = torch.cat((gt, labels), 0)
            pred_global = torch.cat((pred_global, outputs.data), 0)

            # get predictions and true values for each item in batch
            for j in range(0, batch_size[0]):
                thisrow = {}
                bi_thisrow = {}
                truerow = {}

                truerow["path"] = item[j]
                thisrow["path"] = item[j]
                if mode == "test":
                    bi_thisrow["path"] = item[j]

                    # iterate over each entry in prediction vector; each corresponds to
                    # individual label
                for k in range(len(CLASS_NAMES)):
                    thisrow["prob_" + CLASS_NAMES[k]] = probs[j, k]
                    truerow[CLASS_NAMES[k]] = true_labels[j, k]

                    if mode == "test":
                        if probs[j, k] >= thrs[k]:
                            bi_thisrow["bi_" + CLASS_NAMES[k]] = 1
                        else:
                            bi_thisrow["bi_" + CLASS_NAMES[k]] = 0
            
                # pred_df = pred_df.append(thisrow, ignore_index=True)
                # true_df = true_df.append(truerow, ignore_index=True)
                thisrow = pd.DataFrame([thisrow])
                truerow = pd.DataFrame([truerow])
                pred_df = pd.concat([pred_df, thisrow], ignore_index=True)
                true_df = pd.concat([true_df, truerow], ignore_index=True)
                if mode == "test":
                    # bi_thisrow = pd.DataFrame.from_dict(bi_thisrow)
                    # bi_pred_df = bi_pred_df.append(bi_thisrow, ignore_index=True)
                    bi_thisrow = pd.DataFrame([bi_thisrow])
                    bi_pred_df = pd.concat([bi_pred_df, bi_thisrow], ignore_index=True)

        if mode == "test":
            AUROCs_g = compute_AUCs(gt, pred_global)
            AUROC_avg = np.array(AUROCs_g).mean()
            logging.info('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
            for i in range(len(CLASS_NAMES)):
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
                    label="{0} (area = {1:0.3f})".format(label, roc_auc[i]),
                )
            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("{}".format(args.model))
            plt.legend(loc="lower right")
            plt.savefig(args.output_dir + 'roc_' + '{}.png'.format(args.model))

        for column in true_df:
            if column not in CLASS_NAMES:  # 第一列path列，跳过，只读取14种疾病的数值
                continue
            actual = true_df[column]  # 每种疾病的真实值
            pred = pred_df["prob_" + column]  # 每种疾病的预测值

            thisrow = {}
            thisrow['label'] = column

            if mode == "test":
                bi_pred = bi_pred_df["bi_" + column]
                thisrow['auc'] = np.nan
                thisrow['auprc'] = np.nan
            else:
                thisrow['bestthr'] = np.nan

            if mode == "test":
                thisrow['auc'] = roc_auc_score(actual.to_numpy().astype(int), pred.to_numpy())
                thisrow['auprc'] = average_precision_score(actual.to_numpy().astype(int), pred.to_numpy())
            else:
                p, r, t = precision_recall_curve(actual.to_numpy().astype(int), pred.to_numpy())
                # Choose the best threshold based on the highest F1 measure
                f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
                bestthr = t[np.where(f1 == max(f1))]

                thrs.append(bestthr)
                thisrow['bestthr'] = bestthr[0]

            if mode == "Threshold":
                thisrow = pd.DataFrame([thisrow])
                Eval_df = pd.concat([Eval_df, thisrow], ignore_index=True)

            if mode == "test":
                thisrow = pd.DataFrame([thisrow])
                TestEval_df = pd.concat([TestEval_df, thisrow], ignore_index=True)

        pred_df.to_csv(args.output_dir + "preds.csv", index=False)
        true_df.to_csv(args.output_dir + "True.csv", index=False)

        if mode == "Threshold":
            Eval_df.to_csv(args.output_dir + "Threshold.csv", index=False)

        if mode == "test":
            TestEval_df.to_csv(args.output_dir + "TestEval.csv", index=False)
            bi_pred_df.to_csv(args.output_dir + "bipred.csv", index=False)

    logging.info("AUC ave: {}".format(TestEval_df['auc'].sum() / 14.0))
    logging.info("done")

    return pred_df, bi_pred_df, TestEval_df


if __name__ == '__main__':
    test()