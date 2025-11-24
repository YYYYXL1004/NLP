import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime

# ==========================================
# 1. 配置日志 (Logging Setup)
# ==========================================
# 创建 logs 文件夹防止报错
os.makedirs("logs", exist_ok=True)

# 获取当前时间，作为日志文件名，避免覆盖
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/train_record_{current_time}.log"

# 配置 logging
# 同时输出到文件(FileHandler)和控制台(StreamHandler)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 文件输出
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s')) # 控制台只看简洁信息
logger.addHandler(console_handler)

logger.info(f"实验开始。日志文件保存在: {log_filename}")

# ==========================================
# 2. 数据与模型准备 (Data & Model)
# ==========================================
def generate_data(batch_size):
    # x: [1.00, 1.01], y: [0.50, 0.51]
    x = torch.rand(batch_size, 1) * (1.01 - 1.00) + 1.00
    y = torch.rand(batch_size, 1) * (0.51 - 0.50) + 0.50
    inputs = torch.cat([x, y], dim=1)
    targets = torch.sin(x * y)
    return inputs, targets

train_inputs, train_targets = generate_data(1000)
test_inputs, test_targets = generate_data(200)

class LinearFitModel(nn.Module):
    def __init__(self):
        super(LinearFitModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearFitModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ==========================================
# 3. 训练循环 (Training with Better Logging)
# ==========================================
epochs = 800
loss_history = [] 

logger.info("开始训练模型...")

# --- 新增：先看一眼没训练时的 Loss 是多少 ---
# 这样你就能看到它从多少降到了 0.00000045
model.eval() # 临时切到评估模式算一次
with torch.no_grad():
    initial_outputs = model(train_inputs)
    initial_loss = criterion(initial_outputs, train_targets)
logger.info(f'Epoch [0/{epochs}] (Init), Loss: {initial_loss.item():.8f}')
model.train() # 记得切回训练模式
# ----------------------------------------

for epoch in range(epochs):
    outputs = model(train_inputs)
    loss = criterion(outputs, train_targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_val = loss.item()
    loss_history.append(loss_val)

    if (epoch + 1) % 20 == 0:
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_val:.8f}')

logger.info("训练完成。")

# ==========================================
# 4. 评估与记录结果 (Evaluation)
# ==========================================
model.eval()
with torch.no_grad():
    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, test_targets)
    
    a = model.linear.weight[0, 0].item()
    b = model.linear.weight[0, 1].item()
    c = model.linear.bias.item()

    # 使用 logger 记录最终结果
    logger.info("="*30)
    logger.info("评估结果 (Evaluation Results)")
    logger.info("="*30)
    logger.info(f"测试集 Loss (MSE): {test_loss.item():.8f}")
    logger.info(f"拟合参数: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
    logger.info(f"拟合方程: z = {a:.4f}x + {b:.4f}y + {c:.4f}")

# ==========================================
# 5. 画图并保存 (Plotting)
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# 保存图片到 logs 文件夹
plot_filename = f"logs/loss_curve_{current_time}.png"
plt.savefig(plot_filename)
logger.info(f"Loss 曲线图已保存至: {plot_filename}")