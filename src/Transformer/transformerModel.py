from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling,AutoModelForCausalLM,AutoTokenizer
import os
# 路径设置
model_name = "uer/gpt2-chinese-cluecorpussmall"
data_path = "Data/chinese_poems.txt"  # Kaggle 中文古诗数据
output_dir = "Model/transformer_finetuned"  # 模型保存路径

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 避免 padding 报错

# 构建文本数据集（字符级）
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=data_path,
    block_size=32  # 每个训练样本最大长度
)

# 设置数据collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,  # 只训练1轮
    per_device_train_batch_size=2,  # 更小batch
    save_steps=10000,  # 只保存一次
    save_total_limit=1,
    logging_steps=20,
    prediction_loss_only=True,
    logging_dir=os.path.join(output_dir, "logs"),
    max_steps=1000  # 只训练1000步，调试用
)

# 训练器初始化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 保存模型和tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Transformer模型已微调完成并保存到：{output_dir}")
