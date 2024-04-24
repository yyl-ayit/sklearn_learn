from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from MyData import MyData


if __name__ == '__main__':
    address_bees = address_ants = r"D:\桌面\日常练习\深度学习\蚂蚁蜜蜂数据\my_data\train"
    label_ants = "ants"
    mydata_ants = MyData(address_ants, label_ants)
    im, label = mydata_ants[0]

    #转换成tensor类型
    tensor_trans = transforms.ToTensor()
    tensor_img = tensor_trans(im)

    #主要的一些参数
    # transforms.Resize
    # transforms.Compose

    writer = SummaryWriter("logs")
    writer.add_image("tensor_img", tensor_img)
    writer.close()
    # print(tensor_img)
