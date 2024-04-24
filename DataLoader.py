from multiprocessing import freeze_support

import torchvision.datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    freeze_support()
    test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    step = 1
    for data in test_loader:
        ims, targets = data
        # print(type(ims))
        # print(type(targets)) #class 'torch.Tensor'
        if step == 1:
            break

    