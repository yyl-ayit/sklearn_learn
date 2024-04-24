import torchvision

#转换为tensor类型
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root = "./dataset", train = True, download = False)#默认为PIL类型
test_set = torchvision.datasets.CIFAR10(root = "./dataset", train = False, download = False)

img, label = train_set[0]

print(label)
print(train_set.classes)