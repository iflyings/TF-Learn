import evaluate
import psutil
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer, pipeline

######################################################################
checkpoint = "bert-base-uncased"
# 完型填空
# pipe = pipeline("fill-mask", model=checkpoint)
# result = pipe("I've been waiting for a HuggingFace course my whole life, I [MASK] this so much!")
# print(f">>> {result}")

# 情感分类
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2) # 文本分类
# inputs = tokenizer("This is the first sentence.", return_tensors="pt") # 编码输入
# outputs = model(**inputs) # 生成文本
# predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
# print(f">>> {predictions}")
######################################################################
# 加载标准数据库
raw_datasets = load_dataset("glue", "mrpc")
print(f"Dataset: {raw_datasets} ")
print(f"使用的RAM: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

# 将数据库进行编码(添加'input_ids','token_type_ids'和'attention_mask')
tokenized_sentences_1 = raw_datasets["train"]["sentence1"]
tokenized_sentences_2 = raw_datasets["train"]["sentence2"]
tokenized_datasets = tokenizer(tokenized_sentences_1, tokenized_sentences_2, padding=True, truncation=True)
# print(f"1. Tokenized Dataset: {tokenized_datasets.shape}")
# 输入的是dataset格式，而是返回字典，无法同时将大量的数据存到内存
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
# map() 函数中 batched=True, num_proc=8 加速处理
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=8)
# 输入的是字典，返回的也是字典
print(f"2. Tokenized Dataset: {tokenized_datasets.column_names}")
print(f"使用的RAM: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

# 解决句子长度统一的问题
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 训练参数设置, evaluation_strategy="epoch" 表示每个epoch查看一次验证评估
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
train_dataset = tokenized_datasets["train"].select(range(20))
eval_dataset = tokenized_datasets["validation"].select(range(10))
test_dataset = tokenized_datasets["test"].select(range(10))

# 评估
metric = evaluate.load("glue", "mrpc")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(model, training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainOutput = trainer.train()
print(f"Train: {trainOutput}")

# 测试
predictions = trainer.predict(test_dataset)
print(f"Predictions: {predictions.predictions.shape}, {predictions.label_ids.shape}")
preds = np.argmax(predictions.predictions, axis=-1)
for (index, pred) in enumerate(preds):
    print(f"{index} - 预测: {predictions.label_ids[pred]} 实际: {test_dataset['label'][index]}")
