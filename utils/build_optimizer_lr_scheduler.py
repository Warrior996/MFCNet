import torch
import torch.optim as optim
from main import arg_parse
args = arg_parse()

def optimizer_lr_scheduler(model, lr):

    if args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=0.0001)
    elif args.optim == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    elif args.optim == "RAdam":
        optimizer = optim.RAdam(model.parameters(), lr=lr,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    if args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30,
                                                 gamma=0.1, verbose=True)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.1, patience=5, verbose=True)
    elif args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, verbose=True)

    return optimizer, lr_scheduler