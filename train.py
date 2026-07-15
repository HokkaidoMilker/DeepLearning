import torchvision


from torch import  nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# # # 如果train_data_size=10, 训练数据集的长度为: 10
# # print("训练数据集的长度为: {}".format(train_data_size))
# # print("测试数据集的长度为: {}".format(test_data_size))
# # print(f"测试集的长度为{train_data_size}")


train_data = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = DataLoader(test_data, batch_size=64,shuffle=True)




#创建训练模型
tudui = Tudui()

#创建损失函数
loss_fn=  nn.CrossEntropyLoss()


#创建优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

#开始训练
epochs = 10

n=0
total_train_step=  0
total_test_step=0

writer = SummaryWriter("train_log")

for i  in range(epochs):
    print(f"第{i+1}次测试开始")
    for data in train_data:
        img, target = data
        output = tudui(img)
        train_loss = loss_fn(output, target)
        #优化模型
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if total_train_step % 100 == 0:
            print(f"第{total_train_step}次训练的损失为{train_loss.item()}")
            writer.add_scalar("train_loss", train_loss, total_train_step)
        total_train_step += 1

    with torch.no_grad():
        total_test_loss = 0
        total_accuracy = 0
        for data in test_data:
            img, target = data
            output = tudui(img)
            test_loss = loss_fn(output, target)
            total_test_loss += test_loss.item()
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy

        print(f"整体测试集损失为{total_test_loss}")
        print(f"整体测试集的正确率为{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
torch.save(tudui, "./tudui_train.pth")




writer.close()

