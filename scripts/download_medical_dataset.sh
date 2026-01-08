#!/bin/bash
# -*- coding: utf-8 -*-
# 下载医疗数据集脚本
# 数据集来源: https://huggingface.co/datasets/shibing624/medical

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  医疗数据集下载脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/finetune"
REPO_NAME="shibing624/medical"

# 创建数据目录
echo -e "${YELLOW}[1/4] 创建数据目录...${NC}"
mkdir -p "${DATA_DIR}"
echo -e "${GREEN}✓ 数据目录: ${DATA_DIR}${NC}"

# 检查是否安装了 huggingface-cli
echo -e "${YELLOW}[2/4] 检查依赖...${NC}"
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}✗ 未找到 huggingface-cli，正在安装...${NC}"
    pip install -U "huggingface_hub[cli]"
else
    echo -e "${GREEN}✓ huggingface-cli 已安装${NC}"
fi

# 检查是否安装了 git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo -e "${YELLOW}! 未找到 git-lfs，大文件下载可能较慢${NC}"
    echo -e "${YELLOW}  建议安装: https://git-lfs.github.com/${NC}"
fi

# 下载数据集
echo -e "${YELLOW}[3/4] 下载数据集...${NC}"
echo -e "${YELLOW}  来源: ${REPO_NAME}${NC}"

# 方法1: 使用 huggingface-cli 下载 (推荐)
echo -e "${GREEN}使用 huggingface-cli 下载...${NC}"
huggingface-cli download "${REPO_NAME}" \
    --repo-type dataset \
    --local-dir "${DATA_DIR}/medical" \
    --local-dir-use-symlinks False \
    --resume-download

# 备用方法: 使用 git clone (如果上面的命令失败)
# git clone https://huggingface.co/datasets/${REPO_NAME} "${DATA_DIR}/medical"

echo -e "${GREEN}✓ 数据集下载完成${NC}"

# 显示下载的文件
echo -e "${YELLOW}[4/4] 下载结果:${NC}"
if [ -d "${DATA_DIR}/medical/finetune" ]; then
    echo -e "${GREEN}找到以下文件:${NC}"
    ls -lh "${DATA_DIR}/medical/finetune/"
else
    echo -e "${GREEN}找到以下文件:${NC}"
    ls -lh "${DATA_DIR}/medical/"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  数据集下载完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}数据集位置: ${DATA_DIR}${NC}"
echo -e "${YELLOW}使用方法:${NC}"
echo -e "  python supervised_finetuning.py \\"
echo -e "    --model_name_or_path model_path \\"
echo -e "    --train_file_dir ${DATA_DIR}/medical/finetune \\"
echo -e "    --output_dir output/sft_model"
