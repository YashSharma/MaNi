import time
import copy
import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

from infomax_loss import DeepInfoMaxLoss

from utils import dice_score, eval_model, dice_loss, calc_loss, calc_bce_pos

# Find device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hard Code Domain Loss
domain_loss_func = nn.BCEWithLogitsLoss()

def train_model(model, dataloaders, num_iterations=5000, warmup_period=500, domain_w=0., proto_w=0., infomax_loss_type="concat"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0
    
    if (proto_w != 0):
        # infomax loss
        loss_fn = DeepInfoMaxLoss(type=infomax_loss_type).to(device)
        optimizer_fn = optim.Adam(loss_fn.parameters(), lr=1e-4)
    optimizer_flag = 0
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)    

    # Phase
    phase = "train"
    domain_iter = iter(dataloaders['domain'])
    train_iter = iter(dataloaders['train'])

    # Running Loss
    running_loss_domain = 0.0
    running_loss_mask = 0.0
    running_loss_proto = 0.0
    
    # Iterations
    warmup_period = warmup_period
    proto_start_epoch = warmup_period
    for it in range(num_iterations): 
        if (it > warmup_period) & (proto_w != 0):
            if optimizer_flag == 0:
                optimizer_flag = 1
        
        if it % 200 == 0:
            model.eval()
            score = eval_model(dataloaders['val'], model)
            print('Dice Score: {}'.format(score))
            if score > best_score:
                best_score = score
                best_model_wts = copy.deepcopy(model.state_dict())
            model.train()
                        
        # Each epoch has a training and validation phase
        if (domain_w != 0) or (proto_w != 0):
            try:
                inputs_unlab, masks_unlab, _, domain_unlab, domain_tracker_unlab = next(domain_iter)
            except StopIteration:
                domain_iter = iter(dataloaders['domain'])
                inputs_unlab, masks_unlab, _, domain_unlab, domain_tracker_unlab = next(domain_iter)
                
            inputs_unlab = inputs_unlab.to(device)                
            masks_unlab = masks_unlab.to(device, dtype=torch.float)        
            domain_unlab = domain_unlab.to(device)                

        try:
            inputs_lab, masks_lab, _, domain_lab, domain_tracker_lab = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloaders['train'])
            inputs_lab, masks_lab, _, domain_lab, domain_tracker_lab = next(train_iter)

        inputs_lab = inputs_lab.to(device)
        domain_lab = domain_lab.to(device)        
        masks_lab = masks_lab.to(device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()
        if optimizer_flag != 0:
            optimizer_fn.zero_grad()
            
        if (it > warmup_period) & (proto_w != 0):
            logit_label, domain_lab_pred, proto_label_pos, proto_label_neg, \
            logit_unlabel, domain_unlab_pred, proto_unlabel_pos, proto_unlabel_neg = model(inputs_lab, inputs_unlab,\
                                                                                          x_label_map=masks_lab)
        else:        
            if (it>warmup_period) & (domain_w != 0):
                # forward
                # track history if only in train
                logit_label, domain_lab_pred, proto_label_pos, proto_label_neg, \
                logit_unlabel, domain_unlab_pred, proto_unlabel_pos, proto_unlabel_neg = model(inputs_lab, inputs_unlab)
            else:
                logit_label, domain_lab_pred, proto_label_pos, proto_label_neg = model(inputs_lab)
        
        loss = 0
        
        loss_mask = calc_loss(logit_label, masks_lab)
        loss = loss + 1.*loss_mask
        if it > warmup_period:
            if (domain_w != 0):
                loss_domain_lab = domain_loss_func(domain_lab_pred, domain_lab) 
                loss = loss + domain_w*loss_domain_lab
                loss_domain_unlab = domain_loss_func(domain_unlab_pred, domain_unlab) 
                loss = loss + domain_w*loss_domain_unlab
                
            if (proto_w != 0):
                loss_proto = loss_fn(proto_label_pos, proto_label_neg, proto_unlabel_pos)
                loss = loss + proto_w*loss_proto
                                
        loss.backward()
        optimizer.step()
        if optimizer_flag != 0:
            optimizer_fn.step()
                        
        # statistics
        running_loss_mask += loss_mask.item() * inputs_lab.size(0)
        if it > warmup_period:
            if (domain_w != 0):
                running_loss_domain += loss_domain_lab.item() * inputs_lab.size(0)                
                running_loss_domain += loss_domain_unlab.item() * inputs_unlab.size(0) 
            if (proto_w != 0):
                running_loss_proto += loss_proto.item() * inputs_lab.size(0)                                
        
        if it % 50 == 0:
            epoch_loss_mask = running_loss_mask / 50
            epoch_loss_domain = running_loss_domain / 100
            epoch_loss_proto = running_loss_proto / 50
            print('Proto Loss: {}'.format(epoch_loss_proto))
            running_loss_domain = 0.0
            running_loss_mask = 0.0
            running_loss_proto = 0.0

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Score: {:4f}'.format(best_score))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model