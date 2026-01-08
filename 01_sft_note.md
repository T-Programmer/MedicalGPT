## 数据集格式要求

### 文件格式

* **格式** : `.json` 或 `.jsonl`
* **编码** : UTF-8
* **位置** : 放在指定目录，如 `data/finetune/`

### JSON 结构

 **基本格式（单轮对话）** :

```json
{
  "conversations": [
    {"from": "human", "value": "用户问题"},
    {"from": "gpt", "value": "模型回答"}
  ]
}
```

 **多轮对话** :

```json
{
  "conversations": [
    {"from": "human", "value": "第一个问题"},
    {"from": "gpt", "value": "第一个回答"},
    {"from": "human", "value": "第二个问题"},
    {"from": "gpt", "value": "第二个回答"}
  ]
}
```

 **带系统提示词** :

```json
{
  "system_prompt": "你是一个专业的医疗助手",
  "conversations": [
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "回答"}
  ]
}
```

 **或使用 system 角色** :

```json
{
  "conversations": [
    {"from": "system", "value": "你是一个专业的医疗助手"},
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "回答"}
  ]
}
```


# 数据集转换

## 中文数据集

```bash
# 中文训练集 (1,949,972 条)
python convert_dataset.py \
  --in_file mydata/medical/finetune/train_zh_0.json \
  --out_file mydata/medical/finetune/train_zh_0_sharegpt.jsonl \
  --data_type alpaca \
  --file_type json

# 中文验证集
python convert_dataset.py \
  --in_file mydata/medical/finetune/valid_zh_0.json \
  --out_file mydata/medical/finetune/valid_zh_0_sharegpt.jsonl \
  --data_type alpaca \
  --file_type json

# 中文测试集
python convert_dataset.py \
  --in_file mydata/medical/finetune/test_zh_0.json \
  --out_file mydata/medical/finetune/test_zh_0_sharegpt.jsonl \
  --data_type alpaca \
  --file_type json
```

## 英文数据集

```bash
# 英文训练集 (116,617 条)
python convert_dataset.py \
  --in_file mydata/medical/finetune/train_en_1.json \
  --out_file mydata/medical/finetune/train_en_1_sharegpt.jsonl \
  --data_type alpaca \
  --file_type json

# 英文验证集
python convert_dataset.py \
  --in_file mydata/medical/finetune/valid_en_1.json \
  --out_file mydata/medical/finetune/valid_en_1_sharegpt.jsonl \
  --data_type alpaca \
  --file_type json

# 英文测试集
python convert_dataset.py \
  --in_file mydata/medical/finetune/test_en_1.json \
  --out_file mydata/medical/finetune/test_en_1_sharegpt.jsonl \
  --data_type alpaca \
  --file_type json
```