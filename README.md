# 操作步骤
## （一）单独测试RNN模型
1.按照RNNModel-train_rnn-generate的顺序执行，在命令行输入
```
 & generate_rnn.py的绝对路径 --start "春" --length 50 --top_k 5 --model_path "Model/rnn_model.pt"
```
## （二）单独测试Transformer
还是先训练再使用