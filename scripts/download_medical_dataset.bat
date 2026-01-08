@echo off
REM -*- coding: utf-8 -*-
REM 下载医疗数据集脚本 (Windows)
REM 数据集来源: https://huggingface.co/datasets/shibing624/medical

setlocal enabledelayedexpansion

echo ========================================
echo   医疗数据集下载脚本
echo ========================================

REM 获取项目根目录
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "DATA_DIR=%PROJECT_ROOT%\data\finetune"

REM 步骤1: 创建数据目录
echo.
echo [1/4] 创建数据目录...
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
echo ✓ 数据目录: %DATA_DIR%

REM 步骤2: 检查/安装依赖
echo.
echo [2/4] 检查依赖...
where huggingface-cli >nul 2>nul
if %errorlevel% neq 0 (
    echo ✗ 未找到 huggingface-cli，正在安装...
    pip install -U "huggingface_hub[cli]"
) else (
    echo ✓ huggingface-cli 已安装
)

REM 步骤3: 下载数据集
echo.
echo [3/4] 下载数据集...
echo   来源: shibing624/medical
echo.
echo 使用 huggingface-cli 下载...
huggingface-cli download shibing624/medical --repo-type dataset --local-dir "%DATA_DIR%\medical" --local-dir-use-symlinks False --resume-download

echo.
echo ✓ 数据集下载完成

REM 步骤4: 显示下载结果
echo.
echo [4/4] 下载结果:
if exist "%DATA_DIR%\medical\finetune" (
    echo 找到以下文件:
    dir /b "%DATA_DIR%\medical\finetune"
) else (
    echo 找到以下文件:
    dir /b "%DATA_DIR%\medical"
)

echo.
echo ========================================
echo   数据集下载完成！
echo ========================================
echo.
echo 数据集位置: %DATA_DIR%
echo.
echo 使用方法:
echo   python supervised_finetuning.py --model_name_or_path model_path --train_file_dir %DATA_DIR%\medical\finetune --output_dir output\sft_model
echo.

pause
