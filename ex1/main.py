import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def plot_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig("loss_curve.png")
    print("Loss 曲线图已保存至: loss_curve.png")
    plt.close()

def generate_data(batch_size):
    x = torch.rand(batch_size, 1) * (1.01 - 1.00) + 1.00
    y = torch.rand(batch_size, 1) * (0.51 - 0.50) + 0.50
    inputs = torch.cat([x, y], dim=1)
    targets = torch.sin(x * y)
    return inputs, targets

class LinearFitModel(nn.Module):
    def __init__(self):
        super(LinearFitModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

def train_model(model, inputs, targets, epochs=800, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []
    
    print("开始训练模型...")
    
    for epoch in range(epochs):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_val:.8f}')
            
    print("训练完成。")
    return loss_history

def evaluate_model(model, inputs, targets):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        w = model.linear.weight
        b = model.linear.bias
        a_val, b_val = w[0, 0].item(), w[0, 1].item()
        c_val = b.item()

        print("="*30)
        print("评估结果")
        print("="*30)
        print(f"测试集 Loss (MSE): {loss.item():.8f}")
        print(f"拟合参数: a = {a_val:.4f}, b = {b_val:.4f}, c = {c_val:.4f}")
        print(f"拟合方程: z = {a_val:.4f}x + {b_val:.4f}y + {c_val:.4f}")

def main():
    train_inputs, train_targets = generate_data(1000)
    test_inputs, test_targets = generate_data(200)
    
    model = LinearFitModel()
    loss_history = train_model(model, train_inputs, train_targets, epochs=800, lr=0.01)
    evaluate_model(model, test_inputs, test_targets)
    plot_loss(loss_history)

if __name__ == "__main__":
    main()
