#!/bin/bash

# 藏文分词器启动脚本
# 作者: CodeBuddy
# 日期: 2025/8/19

echo "正在启动藏文分词器..."

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请安装Python3后再运行此脚本"
    exit 1
fi

# 检查必要的文件是否存在
if [ ! -f "TibetanTokenizerGUI.py" ]; then
    echo "错误: 未找到TibetanTokenizerGUI.py文件"
    exit 1
fi

if [ ! -d "model" ]; then
    echo "错误: 未找到model目录"
    exit 1
fi

if [ ! -f "model/NyimaTashi.pkl" ] || [ ! -f "model/ti_datasave.pkl" ]; then
    echo "错误: 模型文件不完整，请确保model目录下有NyimaTashi.pkl和ti_datasave.pkl文件"
    exit 1
fi

# 检查并安装依赖
echo "检查依赖..."
python3 -c "import tkinter, PIL, torch" 2>/dev/null || {
    echo "正在安装必要的依赖..."
    pip3 install -r requirments.txt
}

# 启动程序
echo "启动藏文分词器界面..."
python3 TibetanTokenizerGUI.py

echo "程序已退出"