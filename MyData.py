import os

from PIL import Image
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self,address_file,label):
        self.__address_file = address_file
        self.__label = label

        self.__address_relative = os.path.join(address_file,label)
        self.__file_names = []
        for root, dirs, files in os.walk(self.__address_relative):
            for file in files:
                self.__file_names.append(os.path.join(root,file))


    def __getitem__(self, item):
        if isinstance(item, slice):
            files = self.__file_names[item.start: item.stop: item.step]
            img = []
            for file in files:
                img.append(Image.open(file))
            return img, self.__label
        else:
            file = self.__file_names[item]
            img = Image.open(file)
            return img, self.__label

    def __len__(self):
        return len(self.__file_names)

    def __str__(self):
        return "[ants,bees]"

if __name__ == '__main__':
    address_bees=address_ants = r"D:\桌面\日常练习\深度学习\蚂蚁蜜蜂数据\my_data\train"
    label_ants = "ants"
    mydata_ants = MyData(address_ants, label_ants)
    label_bees = "bees"
    mydata_bees = MyData(address_bees, label_bees)

    print(len(mydata_ants))
    # im, label = mydata_ants[0:2]
    # im[1].show()
    print(mydata_bees)