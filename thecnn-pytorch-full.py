import os#从文件导入torch
#import cv2#首先预处理照片：面部识别和裁切图像尺寸
import pandas as pd#科学计算
#import numpy as np
import torch#深度学习模型
from torch.utils.data import Dataset, DataLoader#给torch做好每组输入的内容
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN#更高级的人脸识别库
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm#进度显示





# 检查是否有可用的 GPU
if torch.cuda.is_available():
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    # 打印当前 GPU 的名称
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Using GPU: {gpu_name}")







target_size=(224,224)

mtcnn = MTCNN(device=device)# 初始化 MTCNN 检测器，并将其移动到 GPU
def detect_and_crop_face(image):# 面部检测+尽可能等比例填充
    # 检测图像中的人脸
    boxes, _ = mtcnn.detect(image)
    
    if boxes is not None and len(boxes) > 0:
        # 如果检测到人脸，裁剪并返回第一张人脸
        x1, y1, x2, y2 = boxes[0]
        image = image.crop((x1, y1, x2, y2))
        scale = max(target_size[0] / image.width, target_size[1] / image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        # 中心裁剪
        left = (image.width - target_size[0]) / 2
        top = (image.height - target_size[1]) / 2
        right = (image.width + target_size[0]) / 2
        bottom = (image.height + target_size[1]) / 2
        image = image.crop((left, top, right, bottom))
        return image
    
    # 如果未检测到人脸，则返回 None
    return image




# 定义自定义数据集
class BMI_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)#csv的列数
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, -1])# 获取图片路径(-1代表最后一列，可以用'R'代替)
        image = Image.open(img_name).convert('RGB')  # 打开图片并转换为RGB
        image = detect_and_crop_face(image)# 面部检测和裁剪
        bmi = self.data_frame.iloc[idx, -3]  # 获取BMI值
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(bmi, dtype=torch.float32)#创建张量,32位浮点。

# 数据集路径
csv_file = r'D:\A1-CV-DATABASE\data\train.csv'
root_dir = r'D:\A1-CV-DATABASE\data\face'#脸的路径

# 定义数据增强和预处理(现在改成了图像数据传递)
transform = transforms.Compose([
     transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 实例化数据集和数据加载器
dataset = BMI_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)#一次32个样本（图像及BMI）+打乱。

# 测试数据加载器
#for images, bmis in data_loader:
    #print(images.shape)  # 输出图片的张量形状
    #print(bmis)  # 输出BMI值
    #break  # 仅打印第一个batch的信息
    
    


# 定义 SimpleCNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1)  # 输出一个值，用于预测BMI
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型并将其移动到 GPU
model = SimpleCNN().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

num_epochs = 10  # 设置训练的轮数

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    
    # 使用 tqdm 包装 data_loader，显示训练进度
    for images, bmis in tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images = images.to(device)  # 将图像数据移动到 GPU
        bmis = bmis.to(device)  # 将 BMI 标签移动到 GPU
        
        optimizer.zero_grad()  # 清空梯度
        
        outputs = model(images)  # 前向传播
        outputs = outputs.squeeze()  # 输出从 (batch_size, 1) 调整为 (batch_size,)
        
        loss = criterion(outputs, bmis)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')
    


# 定义保存路径
model_save_path = r'C:\Users\Hasee\testcv.pth'

# 保存模型
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


