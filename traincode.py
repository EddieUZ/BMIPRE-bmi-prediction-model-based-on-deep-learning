import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

# 初始化列表来保存损失值
train_losses = []
valid_losses = []

# 初始化 MTCNN 检测器
mtcnn = MTCNN()
target_size = (240, 320)

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Using GPU: {gpu_name}")
else:
    device = torch.device("cpu")


def detect_and_crop_face(image, target_size=(240, 320)):
    # 使用 MTCNN 进行人脸检测
    boxes, _ = mtcnn.detect(image)

    if boxes is not None and len(boxes) > 0:
        # 获取第一个人脸的坐标
        x1, y1, x2, y2 = boxes[0]

        # 计算扩展边界的偏移量，扩展20%
        width = x2 - x1
        height = y2 - y1
        offset_x = width * 0.2  # 向外扩展20%
        offset_y = height * 0.2  # 向外扩展20%

        # 扩展人脸区域
        x1 = max(x1 - offset_x, 0)  # 确保坐标不小于0
        y1 = max(y1 - offset_y, 0)  # 确保坐标不小于0
        x2 = x2 + offset_x
        y2 = y2 + offset_y
        
        # 裁剪出人脸区域
        face = image.crop((x1, y1, x2, y2))  # 裁剪人脸区域

        # 计算保持比例的缩放
        scale = min(target_size[0] / face.width, target_size[1] / face.height)
        new_size = (int(face.width * scale), int(face.height * scale))
        face = face.resize(new_size, Image.Resampling.LANCZOS)

        # 如果人脸太小，使用 padding 保持目标尺寸
        left = (face.width - target_size[0]) / 2 if face.width > target_size[0] else 0
        top = (face.height - target_size[1]) / 2 if face.height > target_size[1] else 0
        right = (face.width + target_size[0]) / 2 if face.width > target_size[0] else face.width
        bottom = (face.height + target_size[1]) / 2 if face.height > target_size[1] else face.height

        face = face.crop((left, top, right, bottom))

        # 如果目标区域不足，进行填充
        face = face.resize(target_size, Image.Resampling.LANCZOS)
        plt.imshow(face)
        return face
    else:
        return Image.new('RGB', target_size, (0, 0, 0))  # 返回黑色图像作为替代


# 自定义数据集类
class BMI_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, -1])
        image = Image.open(img_name).convert('RGB')
        image = detect_and_crop_face(image)
        bmi = self.data_frame.iloc[idx, -3]  # 获取BMI值
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(bmi, dtype=torch.float32)


# 数据集路径
csv_file = r'D:\A1-CV-DATABASE\data\train.csv'
root_dir = r'D:\A1-CV-DATABASE\data\face'


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 实例化数据集和数据加载器
dataset = BMI_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 验证集数据加载器
valid_dataset = BMI_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform)  # 假设验证集与训练集相同
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 使用预训练的 ResNet18
class BMI_ResNet(nn.Module):
    def __init__(self):
        super(BMI_ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # 修改最后的全连接层
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)

# 实例化模型并将其移动到 GPU
model = BMI_ResNet().to(device)

# 损失函数和优化器
criterion = nn.SmoothL1Loss()  # Huber Loss (SmoothL1Loss) 是 PyTorch 提供的替代选择
optimizer = AdamW(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练轮数
num_epochs = 10  # 增加训练轮数

# 训练过程
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

        # 累加训练损失
        running_loss += loss.item()

    # 计算并记录训练损失
    avg_train_loss = running_loss / len(data_loader)
    train_losses.append(avg_train_loss)

    # 验证过程
    model.eval()  # 设置模型为评估模式
    valid_loss = 0.0
    with torch.no_grad():
        for images, bmis in tqdm(valid_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            bmis = bmis.to(device)

            outputs = model(images)
            outputs = outputs.squeeze()
            loss = criterion(outputs, bmis)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)

    # 更新学习率
    scheduler.step()

    # 打印每个 epoch 的损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')

# 定义保存路径
model_save_path =  r'D:\A1-CV-DATABASE\a.pth'
torch.save(model.state_dict(), model_save_path)
