import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
data_path = "D:\\Practice\\DeepLearnProj\\Data\\chinese_poems.txt"
model_save_path = "D:\\Practice\\DeepLearnProj\\Model\\rnn_model.pt"

# 加载数据
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read().replace('\n', '')

# 构建词表
# chars = list(set(text))
chars = sorted(list(set(text))) 
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(char2idx)

# 数据编码为索引序列
data_tensor = torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)

# 批量获取函数
def get_batch(data, seq_len, batch_size):
    starts = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x_batch = torch.stack([data[i:i+seq_len] for i in starts])
    y_batch = torch.stack([data[i+1:i+seq_len+1] for i in starts])
    return x_batch, y_batch

# 定义LSTM生成器模型
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super(LSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        if hidden is None:
            output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

# 模型与优化器
model = LSTMGenerator(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.CrossEntropyLoss()

# 训练参数
seq_len = 50
batch_size = 32
epochs = 100

# 开始训练
print("开始训练...")
loss_list = []
for epoch in range(1, epochs + 1):
    model.train()
    x_batch, y_batch = get_batch(data_tensor, seq_len, batch_size)
    logits, _ = model(x_batch)
    loss = loss_fn(logits.view(-1, vocab_size), y_batch.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

# 绘制损失曲线
plt.plot(range(1, epochs + 1), loss_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig("Visual/loss_curve.png")
plt.show()

# 保存模型
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存至 {model_save_path}")
