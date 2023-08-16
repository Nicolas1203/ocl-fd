import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

class CORe50Session(ImageFolder):
    def __init__(self, root, train, transform, session_id):
        if session_id in [3, 7, 10] and not train:
            Warning('Session id must be 3, 7 or 10 for testing and oppositely for training.')
        super().__init__(os.path.join(root, f'core50/train/s{session_id}' if train else f'core50/test/s{session_id}'), transform)
        self.transform_in = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size=(128,128))
        ])
    
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = self.transform_in(img)
        label = torch.tensor(label)
        return img, label