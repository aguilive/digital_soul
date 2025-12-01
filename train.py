import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 检查有没有“独显” (GPU)
# 这就是计算机组成里的“硬件加速”。如果有 GPU，训练速度会快几十倍。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# 2. 准备数据 (教材)
# 我们需要把图片转换成 Tensor (张量)，并归一化 (让数值在 0-1 之间，方便计算)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST 数据集的均值和标准差，这是经验值
])

# 下载训练集 (60000张) 和 测试集 (10000张)
# 第一次运行时会自动下载，可能会花点时间
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 制作“数据加载器” (DataLoader)
# 它负责把数据打包，一批一批(Batch)地喂给 GPU，就像流水线进料口
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 3. 搭建神经网络 (大脑结构)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 1: 就像人的眼睛，提取边缘特征
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) 
        # 卷积层 2: 提取更复杂的形状特征
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 丢弃层 (Dropout): 随机关掉一些神经元，防止它“死记硬背”，强迫它学规律
        self.conv2_drop = nn.Dropout2d()
        # 全连接层 1: 汇总特征
        self.fc1 = nn.Linear(320, 50)
        # 全连接层 2: 输出 10 个概率 (对应数字 0-9)
        self.fc2 = nn.Linear(50, 10)

    # 定义数据流通的路径 (Forward Path) —— 你的强项：数据通路
    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2)) # 卷积 -> 池化 -> 激活
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320) # 把二维图片展平成一维向量
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1) # 输出概率

# 初始化模型并搬运到 GPU
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # 优化器：负责修正参数

# 4. 定义训练过程 (上课)
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # 把数据搬到 GPU
        optimizer.zero_grad()   # 清空之前的梯度
        output = model(data)    # 正向传播：猜一下
        loss = torch.nn.functional.nll_loss(output, target) # 计算差距：猜得离谱吗？
        loss.backward()         # 反向传播：计算每个人该背多少锅 (梯度)
        optimizer.step()        # 更新参数：根据梯度修正模型

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# 5. 定义测试过程 (考试)
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 考试时不需要计算梯度，节省显存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # 找出概率最大的那个数字
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\n测试集结果: 平均Loss: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')

# 6. 开始循环训练
# 训练 5 轮 (Epoch)，每一轮都是要把 60000 张图全都看一遍
if __name__ == '__main__':
    for epoch in range(1, 6):
        train(epoch)
        test()
    
    # 7. 保存模型
    # 这就是我们要的“金丹”
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("模型已保存为 mnist_cnn.pth")