import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 加载标准数据库
raw_datasets = load_dataset("glue", "mrpc")
print(f"1. Load Dataset: {raw_datasets}")

# 编码数据库数据
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(f"2. Encode Dataset: {tokenized_datasets}")

# 取出'train'中前面 20 个数据
train_datasets = tokenized_datasets['train'][:20]
column_names = [ v for v, k in train_datasets.items() ]
print(f"3. Train Dataset: {column_names}")

# 'input_ids' 中的数据是否等长
train_datasets_lens = [len(x) for x in train_datasets["input_ids"]]
print(f"4. Train Dataset Legth: {train_datasets_lens}")
print(f"5. Max Train Dataset Legth: {max(train_datasets_lens)}")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 将不需要的列剔除在 data_collator 中无法识别
train_datasets = {k: v for k, v in train_datasets.items() if k not in ["idx", "sentence1", "sentence2"]}
column_names = [ v for v, k in train_datasets.items() ]
print(f"6. Train Dataset Column: {column_names}")
train_padding_datasets = data_collator(train_datasets)
# 验证数据是否对齐了
column_shape = { k : v.shape for k, v in train_padding_datasets.items() }
print(f"7. Train Dataset Length:{column_shape}")