import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = ['benign', 'malignant']

def load_dataset():
    data_path = 'dataset'
    dataset = {}
    for x in ['train', 'test']:
        _dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_path, x),
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )
        dataset[x] = torch.utils.data.DataLoader(
            _dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True
        )

    return dataset

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(100)  # pause a bit so that plots are updated

if __name__ == "__main__":
    inputs, classes = next(iter(load_dataset()['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])