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
