import os
import argparse
import logging
from utils.logging import open_log

def arg_parse():
    nw = os.cpu_count()
    parser = argparse.ArgumentParser(description='ChestX-ray14')
    parser.add_argument('--name', type=str, default="train", help='train, resume, test')
    parser.add_argument('--branch', type=str, default="Fusion", help='Global, Local, Fusion')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--dataset_dir', type=str, default='../../autodl-tmp/ChestX-ray14/images')
    parser.add_argument('--trainSet', type=str, default="./labels/train_list.txt")
    parser.add_argument('--valSet', type=str, default="./labels/val_list.txt")
    parser.add_argument('--testSet', type=str, default="./labels/test_list.txt")
    parser.add_argument('--model', type=str, default="ConvNeXt_Base", help='DenseNet_121, ConvNeXt_base, vision_transformer,'
                                                                            'ResNet50, Res2Net50, ResNet50_ACmix, '
                                                                            'densenet_ACmix, ResNet50_ACmix3, '
                                                                            'densenet_ACmix3, Convnext_base')
    parser.add_argument('--optim', type=str, default="AdamW", help='SGD, Adam, AdamW, RAdam')
    parser.add_argument('--lr_scheduler', type=str, default="ReduceLROnPlateau", help='ReduceLROnPlateau, StepLR, '
                                                                                      'CosineAnnealingLR')
    parser.add_argument('--pretrained', type=bool, default=True, help='True, False')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='True, False')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--output_dir', type=str, default="./results/")
    parser.add_argument('--num_works', type=int, default=15, help='')
    parser.add_argument('--num_epochs', type=int, default=120, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_class', type=int, default=14, help='')
    parser.add_argument('--global_lr', type=float, default=1e-4, help='')
    parser.add_argument('--local_lr', type=float, default=1e-4, help='')
    parser.add_argument('--fusion_lr', type=float, default=1e-4, help='')
    parser.add_argument('--CKPT_PATH_G', type=str, default="results_convnext_global/best_model_path/Best_Global.pkl")
    parser.add_argument('--CKPT_PATH_L', type=str, default="results_convnext_local_vit_seg/best_model_path/Best_Local.pkl")
    parser.add_argument('--CKPT_PATH_F', type=str, default="results/best_model_path/Best_Fusion.pkl")
    parser.add_argument('--UNET_PATH', type=str, default="previous_models/best_unet.pkl")
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--VIT_Seg_PATH', type=str, default="previous_models/VIT_Seg.pth")
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    # gpus = ','.join([str(i) for i in config['GPUs']])
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # open log file
    open_log(args)
    logging.info(args)
    logging.info('Using {} dataloader workers every process'.format(args.num_works))
    if args.branch == 'Global':
        from script.train_model2_global_amp import train
        from script.test_model2_global_amp import test
    elif args.branch == 'Local':
        from script.train_model2_local_Transunet_amp import train
        # from script.train_model2_local_amp import train
        from script.test_model2_local_amp import test
        from script.test_model2_local_Transunet_amp import test
    elif args.branch == 'Fusion':
        from script.train_model2_fusion_Transunet_amp import train
        from script.test_model2_fusion_Transunet_amp import test

    if args.name == 'train':
        train(args)
    elif args.name == 'test':
        test(args)
    elif args.name == 'resume':
        resume(args)

if __name__ == '__main__':
    main()