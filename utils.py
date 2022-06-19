import numpy
import torch
import random
import numpy as np

import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    
def dice_score(pred, target, smooth = 1.):
    
    pred = (torch.sigmoid(pred)>0.5).to(torch.long)

    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    score = (2.*intersection+smooth)/(pred.sum(dim=2).sum(dim=2)+target.sum(dim=2).sum(dim=2)+smooth)
    
    return score.mean()

def eval_model(dl, model):

    model.eval()
    score_tracker = 0
    with torch.no_grad():
        for inputs, masks, _, domain_label, _ in dl:
            inputs = inputs.to(device)
            masks = masks.to(device, dtype=torch.long)
            pred, _, _, _ = model(inputs, validation=True)
            score_tracker += dice_score(pred, masks)
        
    return (score_tracker/len(dl)).item()

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target):

    bce = F.binary_cross_entropy_with_logits(pred, target)
#     bce = 0

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = dice+bce
    
    return loss

def calc_bce_pos(pred, target):
    
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    loss = bce*target
    
    return loss.mean()    