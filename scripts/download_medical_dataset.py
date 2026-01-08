#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载医疗数据集脚本
数据集来源: https://huggingface.co/datasets/shibing624/medical
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("未安装 huggingface_hub，正在安装...")
    os.system("pip install -U huggingface_hub")
    from huggingface_hub import snapshot_download


def print_header(text: str):
    """打印标题"""
    print("\n" + "=" * 50)
    print(f"  {text}")
    print("=" * 50)


def print_step(step: int, total: int, text: str):
    """打印步骤"""
    print(f"\n[{step}/{total}] {text}")


def main():
    # 配置
    REPO_ID = "shibing624/medical"
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "finetune"

    print_header("医疗数据集下载脚本")

    # 步骤1: 创建数据目录
    print_step(1, 4, "创建数据目录...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ 数据目录: {DATA_DIR}")

    # 步骤2: 检查依赖
    print_step(2, 4, "检查依赖...")
    print("✓ huggingface_hub 已安装")

    # 步骤3: 下载数据集
    print_step(3, 4, "下载数据集...")
    print(f"  来源: {REPO_ID}")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(DATA_DIR / "medical"),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("✓ 数据集下载完成")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        sys.exit(1)

    # 步骤4: 显示下载结果
    print_step(4, 4, "下载结果:")

    finetune_dir = DATA_DIR / "medical" / "finetune"
    if finetune_dir.exists():
        print("找到以下文件:")
        for file in sorted(finetune_dir.iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.1f} MB)")
    else:
        medical_dir = DATA_DIR / "medical"
        if medical_dir.exists():
            print("找到以下文件:")
            for file in sorted(medical_dir.iterdir()):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  - {file.name} ({size_mb:.1f} MB)")

    print_header("数据集下载完成！")
    print(f"\n数据集位置: {DATA_DIR}")
    print("\n使用方法:")
    print(f"  python supervised_finetuning.py \\")
    print(f"    --model_name_or_path model_path \\")
    print(f"    --train_file_dir {DATA_DIR}/medical/finetune \\")
    print(f"    --output_dir output/sft_model")


if __name__ == "__main__":
    main()
