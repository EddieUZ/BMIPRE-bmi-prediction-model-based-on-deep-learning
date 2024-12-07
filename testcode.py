#测试124-2顺利plus
import torch
from PIL import Image
from torchvision import transforms
import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from PIL import ImageDraw
# 定义加载模型的路径
model_save_path = r'/testcv_resnet124-2.pth'

# 加载预训练模型
model = BMI_ResNet().to(device)  # 请确保model定义和之前训练时的模型一致
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 设置为评估模式

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  # 将图像转为张量
    
])

def detect_and_crop_face(image, target_size=(240, 320)):
    # 使用 MTCNN 进行人脸检测
    boxes, _ = mtcnn.detect(image)
    if boxes is not None and len(boxes) > 0:
        # 获取第一个人脸的坐标
        x1, y1, x2, y2 = boxes[0]
        
        # 计算扩展边界的偏移量，扩展30%
        width = x2 - x1
        height = y2 - y1
        offset_x = width * 0.3  # 向外扩展30%
        offset_y = height * 0.3  # 向外扩展30%

        # 扩展人脸区域
        x1 = max(x1 - offset_x, 0)  # 确保坐标不小于0
        y1 = max(y1 - offset_y, 0)  # 确保坐标不小于0
        x2 = x2 + offset_x
        y2 = y2 + offset_y

        print(boxes)
        draw = ImageDraw.Draw(image)
        # 绘制矩形框
        for box in boxes:
            # 将box中的坐标转换为整数
            box = [int(coord) for coord in box]
            draw.rectangle(box, outline=(255, 0, 0), width=1)

            # 显示图像
        plt.imshow(image)
        plt.axis('off')  # 不显示坐标轴
        plt.show()
        
        chang=x2-x1
        kuan=y2-y1
        print('横轴长：{:.2f}, 竖轴宽：{:.2f}'.format(chang, kuan))
        
        
        

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
        print("No face detected, returning a blank image.")
        return Image.new('RGB', target_size, (0, 0, 0))  # 返回黑色图像作为替代

# 测试单张图片
def test_single_image(image_path):
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    
    # 应用预处理
    image = detect_and_crop_face(image)
    image = transform(image)
    # 将图像转移到 GPU（如果有）
    image = image.unsqueeze(0).to(device)  # 增加 batch 维度并移至 GPU
    
    # 进行预测
    with torch.no_grad():  # 不需要计算梯度
        output = model(image)
    
    # 输出预测的BMI
    predicted_bmi = output.item()  # 获取单个值
    print(f"Predicted BMI: {predicted_bmi:.4f}")

# 测试图片路径
image_path = r"/usr/3.jpg"

# 调用测试函数
test_single_image(image_path)