import logging
import torch
from model.vit_seg_modeling import VisionTransformer as ViT_seg
from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from main import arg_parse
args = arg_parse()

def ViT_Seg():
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(224 / 16), int(224 / 16))
    model_vit_seg = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes).cuda()
    # load weights
    model_vit_seg.load_state_dict(torch.load(args.VIT_Seg_PATH))
    logging.info("=> loaded well-trained VIT_Seg model checkpoint: {}".format(args.VIT_Seg_PATH))
    model_vit_seg.cuda()
    model_vit_seg.eval()
    return model_vit_seg