"""
本地 bert-base-chinese：加载 → 前向 → Pipeline →（可选）Trainer 微调
数据：ChnSentiCorp 中文酒店评论情感二分类（0 负 / 1 正）
"""
import os
from pathlib import Path

# 下载数据集时走镜像（模型已本地缓存，不依赖此项）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

MODEL_NAME = "bert-base-chinese"
RUN_TRAINER = True
# 学习阶段可先用子集；改成 None 则用全量 train/validation
MAX_TRAIN_SAMPLES = 2000
MAX_EVAL_SAMPLES = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chnsenticorp"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
print(f"已从本地缓存加载: {MODEL_NAME}")

demo_text = "深度学习改变了人工智能。"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

inputs = tokenizer(demo_text, return_tensors="pt").to(model.device)
print("device:", model.device)
print("input_ids:", inputs["input_ids"].shape)
print("tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

with torch.no_grad():
    outputs = model(**inputs)
print("last_hidden_state:", outputs.last_hidden_state.shape)

print("----- Pipeline (feature-extraction) -----")
pipe = pipeline(
    "feature-extraction",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)
feats = pipe(demo_text)
print("pipeline seq_len=", len(feats[0]), "hidden=", len(feats[0][0]))

if RUN_TRAINER:
    print("----- Trainer (ChnSentiCorp) -----")
    raw = load_dataset("lansinuote/ChnSentiCorp")
    train_ds = raw["train"]
    eval_ds = raw["validation"]
    if MAX_TRAIN_SAMPLES is not None:
        train_ds = train_ds.select(range(min(MAX_TRAIN_SAMPLES, len(train_ds))))
    if MAX_EVAL_SAMPLES is not None:
        eval_ds = eval_ds.select(range(min(MAX_EVAL_SAMPLES, len(eval_ds))))
    print("train:", len(train_ds), "eval:", len(eval_ds), "sample:", train_ds[0])

    def tokenize(batch):
        # 不在这里 padding，交给 DataCollator 按 batch 动态补齐
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_tok = train_ds.map(tokenize, batched=True)
    eval_tok = eval_ds.map(tokenize, batched=True)
    cols_to_remove = [c for c in train_tok.column_names if c not in ("input_ids", "attention_mask", "label")]
    train_tok = train_tok.remove_columns(cols_to_remove)
    eval_tok = eval_tok.remove_columns(cols_to_remove)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    clf_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        local_files_only=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(
        model=clf_model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("eval:", metrics)
    trainer.save_model(str(OUTPUT_DIR / "best"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "best"))
    print("已保存到:", OUTPUT_DIR / "best")
