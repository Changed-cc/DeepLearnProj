# rnn_model.py
# 定义了模型结构、加载函数、生成函数
import torch
import torch.nn as nn
import numpy as np

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden
## 加载词表和模型
def load_vocab_and_model(path="D:\\Practice\\DeepLearnProj\\Data\\chinese_poems.txt"
,model_path="D:\\Practice\\DeepLearnProj\\Model\\rnn_model.pt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().replace(" ", "").replace("\n", "")
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    vocab_size = len(char2idx)
    model = LSTMGenerator(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, char2idx, idx2char

def smart_format(poem, words_per_line=7):
    lines = [poem[i:i+words_per_line] for i in range(0, len(poem), words_per_line)]
    return "，\n".join(lines[:-1]) + "。\n" + lines[-1]

def generate(model, char2idx, idx2char, start_char="春", max_len=50, top_k=5):
    input = torch.tensor([[char2idx[start_char]]], dtype=torch.long)
    hidden = None
    result = [start_char]

    for _ in range(max_len):
        logits, hidden = model(input, hidden)
        logits = logits[:, -1, :].squeeze()
        top_k_logits, top_k_idx = torch.topk(logits, k=top_k)
        probs = torch.softmax(top_k_logits, dim=-1)
        next_idx = top_k_idx[torch.multinomial(probs, 1)].item()

        next_char = idx2char[next_idx]
        result.append(next_char)

        input = torch.tensor([[next_idx]], dtype=torch.long)

    return smart_format("".join(result))



