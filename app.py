import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
from torchvision import transforms

# ==========================================
# 1. 定义大脑结构 (必须和训练时一模一样)
# ==========================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ==========================================
# 2. 初始化服务和模型
# ==========================================
app = Flask(__name__)
CORS(app)

# 检查有没有显卡，虽然推理用CPU也飞快，但你有显卡就用显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server running on: {device}")

# 加载模型
model = Net().to(device)
try:
    # 加载你刚才训练好的权重文件
    # map_location确保即使在没有GPU的电脑上也能加载(兼容性写法)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval() # 切换到“考试模式” (关闭 Dropout)
    print(">>> 成功加载模型 mnist_cnn.pth！大脑已这就绪。")
except FileNotFoundError:
    print(">>> ❌ 警告：找不到 mnist_cnn.pth 文件！请确认它和 app.py 在同一个文件夹里。")

# ==========================================
# 3. 定义预处理 (视觉神经)
# ==========================================
# 这里的参数必须和训练时(train.py)保持完全一致
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 转成灰度图 (黑白)
    transforms.Resize((28, 28)),                 # 缩放到 28x28 像素
    transforms.ToTensor(),                       # 转成 Tensor 张量
    transforms.Normalize((0.1307,), (0.3081,))   # 归一化
])


# ... (前面的 import 和 Net 类定义保持不变) ...
# ... (加载模型的部分保持不变) ...

# ==========================================
# 新增：智能裁剪与预处理函数
# ==========================================
def preprocess_image(image):
    # 1. 确保是黑底白字 (RGB转灰度)
    # 如果前端传来的是透明底，先转成黑底
    bg = Image.new("L", image.size, 0) # L表示灰度图，0表示黑色
    if image.mode == 'RGBA':
        # 提取透明通道作为蒙版
        mask = image.split()[3]
        bg.paste(image.convert('L'), mask=mask) 
    else:
        bg.paste(image.convert('L'))
    
    # 2. 获取包围盒 (Bounding Box)
    # getbbox() 会返回非零像素(有笔迹的地方)的坐标 (left, upper, right, lower)
    bbox = bg.getbbox()
    
    if bbox:
        # 如果有笔迹，就裁剪出来
        digit = bg.crop(bbox)
        
        # 3. 填充成正方形 (Padding)
        # 为了不拉伸变形，我们需要把裁剪出来的长方形图，贴到一个正方形黑底的中心
        w, h = digit.size
        max_side = max(w, h)
        
        # 创建一个正方形黑底
        new_im = Image.new("L", (max_side, max_side), 0)
        
        # 把数字贴在正中间
        new_im.paste(digit, ((max_side - w) // 2, (max_side - h) // 2))
        
        # 4. 再次增加一圈边距 (Margin)
        # MNIST 数据集的数字不是顶天立地的，周围有一圈黑边
        # 我们给它加 20% 的黑边
        margin = int(max_side * 0.2)
        final_side = max_side + 2 * margin
        final_im = Image.new("L", (final_side, final_side), 0)
        final_im.paste(new_im, (margin, margin))
        
        # 5. 此时再缩放到 28x28，就非常清晰了
        return final_im.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        # 如果画布是空的，直接返回全黑图
        return bg.resize((28, 28))

# ==========================================
# 修改后的路由函数
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # 1. 解码图片
    image_data_url = request.json['image']
    header, encoded = image_data_url.split(",", 1)
    binary_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(binary_data))
    
    # *** 关键一步：调用智能预处理 ***
    processed_image = preprocess_image(image)
    
    # [调试技巧]：如果你想看 AI 到底看到了什么，把下面这行注释取消掉
    # processed_image.save("debug_input.png") 
    # 这样每次识别，你都能在文件夹里看到一张 saved debug_input.png，看看是不是清晰的数字

    # 2. 转 Tensor 并归一化 (和之前一样)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    input_tensor = transform(processed_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.exp(output)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    print(f"识别结果: {prediction} (置信度: {confidence:.4f})")

    return jsonify({
        "number": prediction,
        "confidence": round(confidence, 4),
        "message": "Success"
    })

# ... (后面的 if __name__ == '__main__': 保持不变)

if __name__ == '__main__':
    # 允许局域网访问
    app.run(port=5000, debug=True)