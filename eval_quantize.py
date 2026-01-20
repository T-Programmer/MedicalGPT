# -*- coding: utf-8 -*-
"""
@description: eval quantize for jsonl format data

usage:
python eval_quantize.py --bnb_path /path/to/your/bnb_model --data_path data/finetune/medical_sft_1K_format.jsonl
"""
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from loguru import logger
import os

parser = argparse.ArgumentParser(description="========量化困惑度测试========")
parser.add_argument(
    "--bnb_path",
    type=str,
    required=True,  # 设置为必须的参数
    help="bnb量化后的模型路径。"
)
parser.add_argument(
    "--data_path",
    type=str,
    required=True,  # 设置为必须的参数
    help="jsonl数据集路径。"
)
parser.add_argument(
    "--use_4bit",
    action="store_true",
    help="是否使用4-bit量化加载模型。"
)


# 设备选择函数
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


# 清理GPU缓存函数
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 从jsonl文件中加载数据
def load_jsonl_data(file_path):
    logger.info(f"Loading data from {file_path}")
    conversations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 提取 human 和 gpt 部分的文本
                for conv in data['conversations']:
                    if conv['from'] == 'human':
                        input_text = conv['value']
                    elif conv['from'] == 'gpt':
                        target_text = conv['value']
                        conversations.append((input_text, target_text))
        return conversations
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []


# 困惑度评估函数
def evaluate_perplexity(model, tokenizer, conversation_pairs):
    def _perplexity(nlls, total_tokens):
        try:
            return torch.exp(torch.stack(nlls).sum() / total_tokens)
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')

    model = model.eval()
    nlls = []
    total_tokens = 0
    # 获取设备
    device = get_device()

    # 确保 tokenizer 和 model 使用相同的设备
    model = model.to(device)

    # 确保 pad_token 存在，如果不存在则使用 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 遍历每个对话，基于 human 部分生成并与 gpt 部分计算困惑度
    for input_text, target_text in tqdm(conversation_pairs, desc="Perplexity Evaluation"):
        # 构建输入：Prompt + Answer
        # 为了计算 PPL，我们需要计算 p(Answer | Prompt)
        # 这里的做法是将 Prompt 和 Answer 拼接，然后 Mask 掉 Prompt 部分的 Loss
        
        # 为了避免粘连，中间加个换行符（视具体模型习惯而定，通用做法加个分隔）
        prompt_text = input_text + "\n"
        
        # 分别编码
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        
        # 拼接 Input IDs
        # 结尾加上 EOS token
        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        
        # 构建 Labels
        # Prompt 部分设为 -100 (忽略 Loss)，Answer 部分保留原值
        labels = [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]
        
        # 截断处理
        max_len = 512
        if len(input_ids) > max_len:
            # 如果过长，从左侧截断（通常保留最新的上下文和完整的回答）
            # 但要注意 labels 和 input_ids 必须同步截断
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]
        
        # 转为 Tensor
        input_tensor = torch.tensor([input_ids]).to(device)
        labels_tensor = torch.tensor([labels]).to(device)

        # Forward pass
        with torch.no_grad():
            # Causal LM 会自动处理 shift，labels 应该与 input_ids 对齐
            outputs = model(input_ids=input_tensor, labels=labels_tensor)
            loss = outputs.loss
            
            # outputs.loss 是标量（平均 Loss），需要还原为 Sum
            # 计算有效的 Token 数量 (label != -100)
            valid_len = (labels_tensor != -100).sum().item()
            
            if valid_len > 0:
                nlls.append(loss * valid_len)
                total_tokens += valid_len

    # 计算最终困惑度
    ppl = _perplexity(nlls, total_tokens)
    logger.info(f"Final Perplexity: {ppl:.3f}")

    return ppl.item()


# 主函数
if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.bnb_path):
        logger.error(f"Model path {args.bnb_path} does not exist.")
        exit(1)

    try:
        # 设置BNB量化配置
        quantization_config = None
        if args.use_4bit:
            from accelerate.utils import BnbQuantizationConfig
            quantization_config = BnbQuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        logger.info(f"Loading model from: {args.bnb_path} (4-bit: {args.use_4bit})")
        tokenizer = AutoTokenizer.from_pretrained(args.bnb_path, use_fast=True, fix_mistral_regex=True)
        model = AutoModelForCausalLM.from_pretrained(args.bnb_path, trust_remote_code=True, quantization_config=quantization_config)

        # 加载jsonl数据
        conversation_pairs = load_jsonl_data(args.data_path)

        if not conversation_pairs:
            logger.error("No valid conversation pairs found.")
            exit(1)

        # 开始评估
        evaluate_perplexity(model, tokenizer, conversation_pairs)

        # 评估完毕，清理模型和缓存
        del model
        clear_gpu_cache()
        logger.info("Evaluation completed and GPU cache cleared.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
