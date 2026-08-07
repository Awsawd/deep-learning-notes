"""
用公开中文三分类语料微调 bert-base-chinese。

默认数据: Kenpache/multilingual-financial-sentiment 中 language=zh
标签: negative=0, neutral=1, positive=2

若 Hub 失败，回退: ChnSentiCorp(正/负) + 内置中性短句弱监督集。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bert-sentiment"
MODEL_NAME = "bert-base-chinese"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_public_3class(max_train: int | None, max_eval: int | None) -> DatasetDict:
    try:
        raw = load_dataset("Kenpache/multilingual-financial-sentiment")
        ds = raw["train"].filter(lambda x: x.get("language") == "zh")

        def map_row(ex):
            lab = str(ex["label"]).lower().strip()
            return {"text": ex["sentence"], "label": int(LABEL2ID[lab])}

        ds = ds.map(map_row, remove_columns=ds.column_names)
        ds = ds.cast_column("label", datasets_class_label())
        split = ds.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
        train, eval_ds = split["train"], split["test"]
        print(f"公开语料(zh) 加载成功: train={len(train)} eval={len(eval_ds)}")
    except Exception as e:
        print("公开三分类加载失败，回退 ChnSentiCorp + 中性弱标签:", e)
        from datasets import Features, Value

        bin_ds = load_dataset("lansinuote/ChnSentiCorp")
        feats = Features({"text": Value("string"), "label": datasets_class_label()})

        def map_bin(ex):
            return {"text": ex["text"], "label": 0 if int(ex["label"]) == 0 else 2}

        train_bin = bin_ds["train"].map(map_bin, remove_columns=bin_ds["train"].column_names)
        eval_bin = bin_ds["validation"].map(map_bin, remove_columns=bin_ds["validation"].column_names)
        train_bin = Dataset.from_list(list(train_bin), features=feats)
        eval_bin = Dataset.from_list(list(eval_bin), features=feats)
        neutrals = [
            "今天天气一般。",
            "我到了会议室。",
            "文件已经发送。",
            "会议改到下周。",
            "请查收附件。",
            "这个问题需要再讨论。",
            "数据如下所示。",
            "以上仅供参考。",
            "已阅。",
            "稍后回复。",
        ] * 200
        neu = Dataset.from_dict({"text": neutrals, "label": [1] * len(neutrals)}, features=feats)
        train = concatenate_datasets([train_bin, neu])
        eval_neu = Dataset.from_dict({"text": neutrals[:100], "label": [1] * 100}, features=feats)
        eval_ds = concatenate_datasets([eval_bin, eval_neu])
        print(f"回退数据: train={len(train)} eval={len(eval_ds)}")

    if max_train is not None:
        train = train.shuffle(seed=42).select(range(min(max_train, len(train))))
    if max_eval is not None:
        eval_ds = eval_ds.shuffle(seed=42).select(range(min(max_eval, len(eval_ds))))
    return DatasetDict({"train": train, "validation": eval_ds})


def datasets_class_label():
    from datasets import ClassLabel

    return ClassLabel(names=["negative", "neutral", "positive"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-eval", type=int, default=800)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--full", action="store_true", help="使用全量（忽略 max-train/eval）")
    args = parser.parse_args()

    max_train = None if args.full else args.max_train
    max_eval = None if args.full else args.max_eval
    raw = load_public_3class(max_train, max_eval)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_tok = raw["train"].map(tokenize, batched=True)
    eval_tok = raw["validation"].map(tokenize, batched=True)
    cols = [c for c in train_tok.column_names if c not in ("input_ids", "attention_mask", "label")]
    train_tok = train_tok.remove_columns(cols)
    eval_tok = eval_tok.remove_columns(cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "runs"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="no",  # 避免存 optimizer 占满磁盘；结束时只存权重
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("eval:", metrics)
    best_dir = OUTPUT_DIR / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print("模型已保存:", best_dir)


if __name__ == "__main__":
    main()
