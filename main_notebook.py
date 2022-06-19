import sys
import cv2
import torch
import albumentations
import pandas as pd
from dataloader import DetectionData
from albumentations.pytorch import ToTensorV2

from load_data import load_data

from model import UNet
from utils import dice_score, eval_model, dice_loss, calc_loss, calc_bce_pos, seed_all
from train import train_model

def run_experiment(domain_w, proto_w, source_domain, target_domain, infomax_loss_type, projection):

    domain_w_flag = float(domain_w)>0
    proto_w_flag = float(proto_w)>0
    
    file_list, valid_file_list, domain_file_list, DOMAIN_KEY, test_file_list = load_data(source=source_domain,
                                                                                        target=target_domain)
    
    seed_all(9)
    
    # Change to 0.75 to mirror old runs
    train_data_transforms = albumentations.Compose([
        albumentations.Flip(p=0.75),
        albumentations.Rotate(p=0.75),
        albumentations.Normalize(0, 1),
        ToTensorV2()
        ])

    valid_data_transforms = albumentations.Compose([
        albumentations.Normalize(0, 1),
        ToTensorV2()
        ])

    train_data = DetectionData(file_list, transform=train_data_transforms, domain=[DOMAIN_KEY])
    valid_data = DetectionData(valid_file_list, transform=valid_data_transforms)
    train_domain_data = DetectionData(domain_file_list, transform=train_data_transforms, domain=[DOMAIN_KEY])    
    
    dataloaders = {'train': torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, drop_last=True),
                   'domain': torch.utils.data.DataLoader(train_domain_data, batch_size=4, shuffle=True, drop_last=True),
                  'val': torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=False)}

    dataset_sizes = {'train': len(train_data), 'val': len(valid_data)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    
    model_ft = UNet(n_channels=3, n_classes=1, domain=domain_w_flag, proto=proto_w_flag, projection=projection)
    model_ft = model_ft.to(device)
    
    model_ft = train_model(model_ft, dataloaders, num_iterations=10000, warmup_period=1000, domain_w=float(domain_w), proto_w=float(proto_w), infomax_loss_type=infomax_loss_type)

    model_ft.eval()
    test_data = DetectionData(test_file_list, transform=valid_data_transforms)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
  
    test_dice_score = eval_model(test_dl, model_ft)
    config.test_dice_score = test_dice_score
    
    file = open("score_source_domain="+source_domain+"_target_domain="+target_domain+"_domain="+domain_w+"_proto="+proto_w+"_loss_type="+infomax_loss_type+"_projection="+projection+".txt", "a")
    file.write(str(test_dice_score)+" \n")
    file.close()   
