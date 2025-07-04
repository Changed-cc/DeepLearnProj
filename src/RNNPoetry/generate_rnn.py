# 命令行入口脚本，用于生成古诗
# | 参数             | 说明                      |
# | -------------- | ----------------------- |
# | `--start`      | 起始字                     |
# | `--length`     | 生成的最大长度（默认为 50）         |
# | `--top_k`      | Top-K 采样参数（默认为 5）       |
# | `--model_path` | 模型路径（默认 `rnn_model.pt`） |

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RnnModel import generate, load_vocab_and_model  # 确保 rnn_model.py 文件中有该函数

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="起始字符")
    parser.add_argument("--length", type=int, default=50, help="生成字符长度")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k 采样参数")
    parser.add_argument("--model_path", type=str, default="rnn_model.pt", help="模型权重路径")

    args = parser.parse_args()

    # 保证模型路径为绝对路径
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

    model, char2idx, idx2char = load_vocab_and_model(model_path=model_path)
    output = generate(model, char2idx, idx2char, start_char=args.start, max_len=args.length, top_k=args.top_k)
    print("\n生成诗句：\n" + output)
