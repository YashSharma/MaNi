import numpy as np
from torch.utils.data import Dataset, DataLoader

class DetectionData(Dataset):
    def __init__(self, file_list, transform=None, domain=None):
        self.file_list = file_list
        self.transform = transform  
        self.domain = domain

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_mask = np.load(self.file_list[idx])
        img = img_mask[:, :, :3].astype('uint8')
        msk = img_mask[:, :, 3].astype('int32')
        if self.transform:
            img = self.transform(image=img, mask=msk)                       
        domain_label = np.zeros(img['mask'].shape)[None]
        domain_tracker = 0
        if self.domain:
            for dm in self.domain:
                if dm in self.file_list[idx]:
                    domain_label = np.ones(img['mask'].shape)[None]                    
                    domain_tracker = 1
        return img['image'], img['mask'][None], self.file_list[idx], domain_label, domain_tracker