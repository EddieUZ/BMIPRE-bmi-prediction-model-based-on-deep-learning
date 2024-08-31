import os#从文件导入torch
#import cv2#首先预处理照片：面部识别和裁切图像尺寸
import pandas as pd#科学计算
#import numpy as np
import torch#深度学习模型
from torch.utils.data import Dataset, DataLoader#给torch做好每组输入的内容
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN#更高级的人脸识别库
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
target_size=(224,224)
mtcnn = MTCNN()# 初始化 MTCNN 检测器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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



# 实例化模型
model = SimpleCNN()
model.load_state_dict(torch.load(r'C:\Users\Hasee\testcv.pth'))
model.to(device)  # 移动模型到 GPU 或 CPU

# 设置模型为评估模式
model.eval()

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



# 定义数据增强和预处理（transform）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224
    transforms.ToTensor(),          # 将图像转换为张量
])

# 之后你可以在代码中使用 transform 进行图像预处理




# 加载和预处理图像
img_path = r'D:\A1-CV-DATABASE\data\test\single_face\me.jpg'
image = Image.open(img_path).convert('RGB')
image = detect_and_crop_face(image)
image = transform(image).unsqueeze(0)  # 添加批次维度

# 移动图像到 GPU 或 CPU
image = image.to(device)

# 预测
with torch.no_grad():
    model.eval()
    output = model(image).item()

print(f'Predicted BMI: {output:.4f}')
