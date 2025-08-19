<h1 align="center">
  <br>
  བོད་ཡིག་གི་ཚིག་མིང་དུ་དབྱེ་བྱེད། | Tibetan Tokenizer | 藏文分词器
  <br>
</h1>

<p align="center">
  <b>基于Bi-LSTM+CRF方法的藏文分词工具</b>
</p>

## 📝 项目简介

本项目是一个高效的藏文分词工具，采用Bi-LSTM+CRF深度学习方法，为藏文自然语言处理研究提供基础支持。工具提供命令行和图形界面两种使用方式，方便不同场景下的应用需求。

## ✨ 功能特点

- 🖥️ **双模式操作**：支持命令行和图形界面两种操作方式
- 📝 **文本处理**：直接在界面中输入藏文进行分词
- 📂 **文件处理**：处理藏文文本文件并保存结果
- 🌐 **多语言支持**：界面支持藏文、英文和中文
- 👨‍💻 **用户友好**：直观的图形界面，操作简单

## 🚀 使用方法

### 环境要求

- Python 3.6+
- 相关依赖包

### 命令行模式

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **执行分词**：
   ```bash
   python NyimaTashi.py <input_file> <output_file>
   ```
   - `<input_file>`: 需要分词的文件路径
   - `<output_file>`: 结果输出文件路径

### 图形界面模式

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **启动图形界面**：
   ```bash
   python TibetanTokenizerGUI.py
   ```

3. **使用方法**：
   - **文本处理**：在左侧输入框中输入藏文文本，点击"分词处理"按钮
   - **文件处理**：切换到文件处理模式，选择文件，点击"分词处理"按钮
   - **保存结果**：点击"保存结果"按钮将分词结果保存到文件
   - **复制结果**：点击"复制结果"按钮将分词结果复制到剪贴板

## 📚 项目结构

- `NyimaTashi.py` - 命令行分词工具
- `TibetanTokenizerGUI.py` - 图形界面分词工具
- `model.py` - 模型定义文件
- `model/` - 预训练模型目录
- `fontfile/` - 字体文件目录

## 🔗 相关链接

原项目地址：[https://github.com/gyatso736/Tibetan-tokenizer-](https://github.com/gyatso736/Tibetan-tokenizer-)
