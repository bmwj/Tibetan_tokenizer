import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox, font
from tkinter.colorchooser import askcolor
import threading
import os
import time
import torch
import re
import pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import model
import itertools
import base64
from PIL import Image, ImageTk
from io import BytesIO

class TibetanTokenizer:
    def __init__(self):
        self.datas_pkl = './model/ti_datasave.pkl'
        self.model_path = './model/NyimaTashi.pkl'
        self.device = torch.device("cpu")
        self.word2id = None
        self.id2word = None
        self.tag2id = None
        self.id2tag = None
        self.load_model_data()

    def load_model_data(self):
        """加载模型数据"""
        try:
            with open(self.datas_pkl, 'rb') as inp:
                self.word2id = pickle.load(inp)
                self.id2word = pickle.load(inp)
                self.tag2id = pickle.load(inp)
                self.id2tag = pickle.load(inp)
            
            START_TAG = "<START>"
            self.tag2id[START_TAG] = len(self.tag2id)
            return True
        except Exception as e:
            print(f"加载模型数据时出错: {e}")
            return False

    def process_text(self, input_text):
        """处理输入文本并返回分词结果"""
        if not self.word2id or not self.id2word:
            return "错误：模型数据未加载"
        
        result_lines = []
        
        for line in input_text.split('\n'):
            if not line.strip():
                result_lines.append('')
                continue
                
            line_last = []
            line = line.strip()
            replacements = {
                'འི': '་འི',
                'འདིར': 'འདི་ར',
                'གྲྭར': 'གྲྭ་ར',
                'པས': 'པ་ས',
                'འང': '་འང',
                'འམ': '་འམ',
                'པོར': 'པོ་ར',
            }
            for old, new in replacements.items():
                line = line.replace(old, new)
            
            line = re.sub('([,.;\':"!@#$%^&*(){}\\[\\]༜༝༼༽༕༖༗ྻ༘༙༚༛༆༇༃༿࿏༾༿༟༾༴,.]+)', r'ヨ\1ヨ', line)
            line = re.sub('།+', '།', line)
            line = line.replace('།', '་།་ ')
            line = re.sub(r'\s+', ' ', line)
            line = re.sub(r'(\d)(?=\d)', r'\1 ', line)
            line = re.sub(r'(\d)(?=\D)|(\D)(?=\d)', r'\1\2 ', line)
            line = re.sub(r'([a-zA_Z]+)|([^a-zA-Z]+)', r'\1 \2', line)
            line = line.lstrip('།')
            line = line.lstrip('་')
            line = line.strip()
            
            pair_nt = re.findall('[^\u0F00-\u0FFF]+', line)
            pair_t = re.findall('[\u0F00-\u0FFF]+', line)
            
            if pair_t:
                processed_line = self.process_line(pair_nt, pair_t, line)
                result_lines.append(processed_line)
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)

    def process_line(self, pair_nt, pair_t, line):
        """处理单行文本"""
        line_last = []
        
        if pair_nt:
            if pair_t[-1] == '' or pair_t[0] == '':
                pair_t = pair_t[:-1]
            
            if line[0] in ' '.join(pair_nt):
                for m in range(len(pair_t)):
                    line_x = []
                    line_s = []
                    terlit = []
                    if '་' in pair_t[m]:
                        line_s = pair_t[m].split('་')
                    else:
                        line_s.append(pair_t[m])
                    
                    for i in range(len(line_s)):
                        if not line_s[i]: continue
                        if (line_s[i] in self.id2word):
                            line_x.append(self.word2id[line_s[i]])
                        else:
                            self.word2id[line_s[i]] = self.word2id['<unk>']
                            terlit.append(line_s[i])
                            line_x.append(self.word2id['<unk>'])
                    
                    if line_x:
                        line_r = self.generate_sentence(line_x)
                        if m < len(pair_nt):
                            if pair_nt[m] == 'ヨ':
                                line_last.append('།')
                            else:
                                line_last.append(pair_nt[m])
                        if len(terlit) != 0:
                            c = itertools.cycle(terlit)
                            line_r = re.sub('<unk>', lambda _: next(c), line_r)
                        line_last.append(line_r)
            else:
                for m in range(len(pair_t)):
                    line_x = []
                    line_s = []
                    terlit = []
                    if '་' in pair_t[m]:
                        line_s = pair_t[m].split('་')
                    else:
                        line_s.append(pair_t[m])
                    
                    for i in range(len(line_s)):
                        if not line_s[i]: continue
                        if (line_s[i] in self.id2word):
                            line_x.append(self.word2id[line_s[i]])
                        else:
                            self.word2id[line_s[i]] = self.word2id['<unk>']
                            terlit.append(line_s[i])
                            line_x.append(self.word2id['<unk>'])
                    
                    if line_x:
                        line_r = self.generate_sentence(line_x)
                        if len(terlit) != 0:
                            c = itertools.cycle(terlit)
                            line_r = re.sub('<unk>', lambda _: next(c), line_r)
                        line_last.append(line_r)
                        if m < len(pair_nt):
                            if pair_nt[m] == 'ヨ':
                                line_last.append('།')
                            else:
                                line_last.append(pair_nt[m])

            line_last = ' '.join(line_last)
            return line_last
        else:
            line = line.strip()
            line_s = []
            line_x = []
            terlit = []
            if '་' in line:
                line_s = line.split('་')
            else:
                line_s.append(line)
            
            for i in range(len(line_s)):
                if not line_s[i]: continue
                if (line_s[i] in self.id2word):
                    line_x.append(self.word2id[line_s[i]])
                else:
                    self.word2id[line_s[i]] = self.word2id['<unk>']
                    line_x.append(self.word2id['<unk>'])
                    terlit.append(line_s[i])
            
            if line_x:
                line_last = self.generate_sentence(line_x)
                if len(terlit) != 0:
                    c = itertools.cycle(terlit)
                    line_last = re.sub('<unk>', lambda _: next(c), line_last)
                return line_last
            return ""

    def generate_sentence(self, sentence):
        """生成分词后的句子"""
        try:
            # 添加安全全局变量声明
            import torch.serialization
            torch.serialization.add_safe_globals(['model.Model'])
            
            # 加载模型，明确指定 weights_only=False
            model_obj = torch.load(self.model_path, map_location=self.device, weights_only=False)
            word_list = [self.id2word[s] for s in sentence]
            sentence_tensor = torch.tensor(sentence, dtype=torch.long)
            _, predict = model_obj.test(sentence_tensor)
            
            list_ = []
            for k, tag in enumerate(predict):
                if tag == 0 or tag == 1:
                    list_.append(word_list[k])
                    list_.append('་')
                else:
                    list_.append(word_list[k])
                    list_.append(' / ')  # 在斜杠两边添加空格
            
            list_l = ''.join(list_)
            return list_l
        except Exception as e:
            print(f"生成分词句子时出错: {e}")
            return "处理错误"

    def process_file(self, input_file, output_file=None):
        """处理文件并保存结果"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_text = f.read()
            
            result = self.process_text(input_text)
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result)
            
            return result  # 返回处理结果，用于在界面上显示
        except Exception as e:
            print(f"处理文件时出错: {e}")
            return None


class IconManager:
    """图标管理类，用于创建和管理按钮图标"""
    
    @staticmethod
    def get_process_icon():
        # 分词处理图标 - 齿轮图标
        icon_data = """
        iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA/0lEQVRIie2UMU7DQBBF3xoLIXEBDkBFQUNNQcMVKKhyAzoKJI7ABWg4ABUFoqbkAJQUkVCCRfFtWBvba6/XTpGfNJI9M/P/7I5mF/7rD2sGOAQGwAQYAjvVKQrWDXABvAIZsABegDPgYBXiTeAOWFokr8AtsNF1gjPgw5F8AqfAHfBhxj+B8y7FD4EXRzQGtoHcGe8Bj2b+Geh3Id4HnkzQFOi5nDUTm5r5CRBVFc+BR0PyAOxG5G+ZnDsgrCKeA/eG4BbYSpA/MrlDIEgVz4BrQc6BnRriXUFyBWQx8Qy4FMQL4KiBeAE8m9wxECwTz4ATQXLT8Pt/VVtfMIVvdwAk8vEAAAAASUVORK5CYII=
        """
        return IconManager._create_image_from_base64(icon_data)
    
    @staticmethod
    def get_clear_icon():
        # 清空图标 - 橡皮擦图标
        icon_data = """
        iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA+ElEQVRIie3UMUoDQRTG8d8qghcQPICNjY2NRxA8gI2ljY2NR7ASGy/gCaxsbCxsLLyAYGEhCGLxJrAsu9nZ7KTwg4Fh5n3z/7KPeab5p9qhxDLwvUKJXZ+JO7zioQbkHu94w3UTgBITzPCEgwjkEFPM8VgHUGKMKR5xFIGcYIEZxjGAEiNMcB+BnOIVn7hqAiixj1vc4SwCOccbvnDTBlBiF9e4xWUEcoEPLHHXBaDEDi5whescSIxaA0qc4wYjnEYgQ3ziB/dtAUrs4QrXGEQgF/jGAvdVAKUJOMcIwwjkEh/4xqQOoMQAI4xxEoEM8W0ydlUHMM0/1S+UzWgVc+l+mAAAAABJRU5ErkJggg==
        """
        return IconManager._create_image_from_base64(icon_data)
    
    @staticmethod
    def get_copy_icon():
        # 复制图标 - 复制图标
        icon_data = """
        iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA2klEQVRIie3UMUpDQRSF4S8qWAQsLKxcgI2laxBcgJWdtZ2VK7GwdQEuwMrOzsoVCBYWFoKF4LW4A8Mj5L2ZvJDAHBiYYc7579zhMcP8U0ussMYHvvCJD7xgUQsyxBRPeMd7i56wqgEZ4BZv2OAYJzjGsZnbYF0KGZjufYsznOPcnGWLexzVQIa4wwuOcIFLXJmz3eABw1LILe7wiitc4wbXZmzXWJZCFuYDvsUdHs3YlqWQoRnLZ9zjHk9mbJ9qbvKvmuEHPzVXVaM+IGt9QNb6gGz1AdnqF76VVh8QlHrAAAAAAElFTkSuQmCC
        """
        return IconManager._create_image_from_base64(icon_data)
    
    @staticmethod
    def get_save_icon():
        # 保存图标 - 保存图标
        icon_data = """
        iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA0ElEQVRIie3UMUpDQRSF4S+KhYVgYWHlAqxtXYOCK7CysrW1chUWti7ABVjZ2dm5AsHCwkKwELwWd2B4hLyZSUDwwIVh5pz/nx/uMMP8U3OssMMnvvGFT7xhWQsyxgqveMdHix6xrQEZ4QFv2OEUJ+YsW7zgpBQywR0+cYYLXOLanO0WjxiXQu7xjBtc4Q73uDFju8G6FLIyH/AW93gyY1uXQkZmLF9Y4RlrM7ZvNTf5V83xg9+aq6rRAGStAchag5C1BiBrDUDW+gNNKVYfVJNkWQAAAABJRU5ErkJggg==
        """
        return IconManager._create_image_from_base64(icon_data)
    
    @staticmethod
    def get_browse_icon():
        # 浏览图标 - 文件夹图标
        icon_data = """
        iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA4ElEQVRIie3UPUpDQRTF8Z8oFhaCRcDCwsoF2Ni6BgUXYGVla2vlKixsXYALsLKzs3MFgoWFhWAheC3uwPCIeTPJg+AcGJhhzv3PmeFOw38tscIan/jCN77wgWUtSB9LvOANHy16xLYGZIAHvGKHU5yYs2zxjJNSyBh3+MApznGBK3O2WzxiWAq5xxNucIl73OPajG2FVSlkZT7gW9zjwYxtXQoZmLF8YoVHrM3YvtXc5F81ww9+a66qRgOQtQYgaw1C1hqArDUAWesXTXNWH1STZFkDkLUGIGsNQNb6BVSjVh9+0YJpAAAAAElFTkSuQmCC
        """
        return IconManager._create_image_from_base64(icon_data)
    
    @staticmethod
    def get_exit_icon():
        # 退出图标 - 退出图标
        icon_data = """
        iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA7UlEQVRIie3UPUpDQRTF8Z8oFhaCRcDCwsoF2Ni6BgUXYGVla2vlKixsXYALsLKzs3MFgoWFhWAheC3uwPCIeTPJg+AcGJhhzv3PmeFOw38tscIan/jCN77wgWUtSB9LvOANHy16xLYGZIAHvGKHU5yYs2zxjJNSyBh3+MApznGBK3O2WzxiWAq5xxNucIl73OPajG2FVSlkZT7gW9zjwYxtXQoZmLF8YoVHrM3YvtXc5F81ww9+a66qRgOQtQYgaw1C1hqArDUAWesXTXNWH1STZFkDkLUGIGsNQNb6BVSjVh9+0YJpAAAAAElFTkSuQmCC
        """
        return IconManager._create_image_from_base64(icon_data)
    
    @staticmethod
    def get_text_mode_icon():
        # 文本模式图标 - 文本图标
        icon_data = """
        iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA1ElEQVRIie3UMUoDQRTG8Z8aQbAQBEEQxMLGI3gEj2BjY2NjYyOewMJDeAQLGxsbGwsPIFgIFoLF4iawLLvZ2eym8IMBmXnf/L/skxnmn1pggy0+8IkvfOIdL1jVgixwjVd84L1Fj7irARnhAc/Y4QQnOMaRmXvCbSlkglu84RTnuMClmbtpIKWQezziBpe4wwOuGkgxZGU+4Fvc4t6MbV0KGZuxfGCFB6zN2D7V3ORfNcU3fmquqkYDkLUGIGsNQtYagKw1AFnrB0OzVh9Uk2RZA5C1fgAXYlYfKHEr0QAAAABJRU5ErkJggg==
        """
        return IconManager._create_image_from_base64(icon_data)
    
    @staticmethod
    def get_file_mode_icon():
        # 文件模式图标 - 文件图标
        icon_data = """
        iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA1ElEQVRIie3UMUoDQRTG8Z8aQbAQBEEQxMLGI3gEj2BjY2NjYyOewMJDeAQLGxsbGwsPIFgIFoLF4iawLLvZ2eym8IMBmXnf/L/skxnmn1pggy0+8IkvfOIdL1jVgixwjVd84L1Fj7irARnhAc/Y4QQnOMaRmXvCbSlkglu84RTnuMClmbtpIKWQezziBpe4wwOuGkgxZGU+4Fvc4t6MbV0KGZuxfGCFB6zN2D7V3ORfNcU3fmquqkYDkLUGIGsNQtYagKw1AFnrB0OzVh9Uk2RZA5C1fgAXYlYfKHEr0QAAAABJRU5ErkJggg==
        """
        return IconManager._create_image_from_base64(icon_data)
    
    @staticmethod
    def _create_image_from_base64(base64_str):
        """从Base64字符串创建图像"""
        base64_data = base64_str.strip()
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
        return ImageTk.PhotoImage(image)

class TibetanTokenizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(" 藏文分词器-བོད་ཡིག་གི་ཚིག་མིང་དུ་དབྱེ་བྱེད།-Tibetan_tokenizer")
        self.root.geometry("1200x800")
        
        # 设置主题颜色
        self.primary_color = "#4a6fa5"  # 主色调
        self.secondary_color = "#f0f5ff"  # 次要色调
        self.accent_color = "#ff7e5f"  # 强调色
        self.text_color = "#333333"  # 文本颜色
        self.bg_color = "#f9f9f9"  # 背景色
        
        # 设置窗口图标和背景
        self.root.configure(bg=self.bg_color)
        
        # 设置字体 - 使用系统支持藏文的字体
        self.title_font = font.Font(family="Arial", size=12, weight="bold")
        self.text_font = font.Font(family="Arial", size=10)
        
        # 尝试加载支持藏文的字体
        tibetan_fonts = ["Microsoft Himalaya", "Kailash", "Jomolhari", "Noto Sans Tibetan", "Arial Unicode MS"]
        self.tibetan_font = None
        
        for font_name in tibetan_fonts:
            try:
                self.tibetan_font = font.Font(family=font_name, size=16)  # 增大字体大小
                break
            except:
                continue
        
        if not self.tibetan_font:
            # 如果没有找到藏文字体，使用默认字体
            self.tibetan_font = font.Font(family="TkDefaultFont", size=16)  # 增大字体大小
        
        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')  # 使用clam主题作为基础
        
        # 配置各种元素样式
        self.style.configure("TButton", 
                            padding=8, 
                            relief="flat", 
                            background=self.primary_color, 
                            foreground="white",
                            font=self.text_font)
        
        self.style.map("TButton",
                      foreground=[('pressed', 'white'), ('active', 'white')],
                      background=[('pressed', self.accent_color), ('active', self.accent_color)])
        
        # 配置退出按钮样式
        self.style.configure("Exit.TButton", 
                           background="#e74c3c", 
                           foreground="white")
        self.style.map("Exit.TButton",
                     foreground=[('pressed', 'white'), ('active', 'white')],
                     background=[('pressed', '#c0392b'), ('active', '#c0392b')])
        
        # 配置模式按钮样式
        self.style.configure("Mode.TButton", 
                           background=self.primary_color, 
                           foreground="white",
                           padding=10,
                           font=self.title_font)
        self.style.map("Mode.TButton",
                     foreground=[('pressed', 'white'), ('active', 'white')],
                     background=[('pressed', self.accent_color), ('active', self.accent_color)])
                     
        # 配置文本模式和文件模式按钮样式
        self.style.configure("TextMode.TButton", 
                           background=self.accent_color, 
                           foreground="white",
                           padding=10,
                           font=self.title_font)
        self.style.configure("FileMode.TButton", 
                           background=self.primary_color, 
                           foreground="white",
                           padding=10,
                           font=self.title_font)
        
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=self.text_font)
        self.style.configure("TLabelframe", background=self.bg_color, foreground=self.text_color, font=self.text_font)
        self.style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.primary_color, font=self.title_font)
        
        # 创建分词器实例
        self.tokenizer = TibetanTokenizer()
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧内容区域和右侧按钮区域
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 调整按钮区域为固定宽度，更合理的尺寸
        self.button_frame = ttk.Frame(self.main_frame, width=180)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        self.button_frame.pack_propagate(False)  # 防止按钮框架被内容压缩
        
        # 创建Unicode图标
        self.icons = {
            'process': "⚙️",
            'clear': "🗑️",
            'copy': "📋",
            'save': "💾",
            'browse': "📂",
            'exit': "🚪",
            'text_mode': "📝",
            'file_mode': "📄"
        }
        
        # 创建模式切换按钮
        self.mode_frame = ttk.Frame(self.button_frame, padding="10")
        self.mode_frame.pack(fill=tk.X, pady=20)
        
        self.text_mode_button = ttk.Button(
            self.mode_frame, 
            text=f"{self.icons['text_mode']} 文本处理", 
            command=self.show_text_mode,
            style="TextMode.TButton",
            width=12  # 设置合适的宽度
        )
        self.text_mode_button.pack(pady=5, padx=2)
        
        self.file_mode_button = ttk.Button(
            self.mode_frame, 
            text=f"{self.icons['file_mode']} 文件处理", 
            command=self.show_file_mode,
            style="FileMode.TButton",
            width=12  # 设置合适的宽度
        )
        self.file_mode_button.pack(pady=5, padx=2)
        
        # 创建操作按钮区域
        self.operation_frame = ttk.LabelFrame(self.button_frame, text="操作", padding="10")
        self.operation_frame.pack(fill=tk.X, pady=10, ipady=10)
        
        # 文本模式按钮 (初始隐藏)
        self.text_operation_frame = ttk.Frame(self.operation_frame)
        
        self.process_text_button = ttk.Button(
            self.text_operation_frame, 
            text=f"{self.icons['process']} 分词处理", 
            command=self.process_text,
            style="TButton",
            width=12
        )
        self.process_text_button.pack(pady=5, padx=2, anchor=tk.CENTER)
        
        self.clear_text_button = ttk.Button(
            self.text_operation_frame, 
            text=f"{self.icons['clear']} 清空文本", 
            command=self.clear_text,
            style="TButton",
            width=12
        )
        self.clear_text_button.pack(pady=5, padx=2, anchor=tk.CENTER)
        
        # 添加复制结果按钮
        self.copy_result_button = ttk.Button(
            self.text_operation_frame, 
            text=f"{self.icons['copy']} 复制结果", 
            command=self.copy_result,
            style="TButton",
            width=12
        )
        self.copy_result_button.pack(pady=5, padx=2, anchor=tk.CENTER)
        
        # 添加退出按钮到操作区域
        self.exit_button_text = ttk.Button(
            self.text_operation_frame, 
            text=f"{self.icons['exit']} 退出程序", 
            command=self.exit_program,
            style="Exit.TButton",
            width=12
        )
        self.exit_button_text.pack(pady=5, padx=2, anchor=tk.CENTER)
        
        # 文件模式按钮 (初始隐藏)
        self.file_operation_frame = ttk.Frame(self.operation_frame)
        
        self.process_file_button = ttk.Button(
            self.file_operation_frame, 
            text=f"{self.icons['process']} 分词处理", 
            command=self.process_file,
            style="TButton",
            width=12
        )
        self.process_file_button.pack(pady=5, padx=2, anchor=tk.CENTER)
        
        self.save_file_button = ttk.Button(
            self.file_operation_frame, 
            text=f"{self.icons['save']} 保存结果", 
            command=self.save_result_file,
            style="TButton",
            width=12
        )
        self.save_file_button.pack(pady=5, padx=2, anchor=tk.CENTER)
        
        self.clear_file_button = ttk.Button(
            self.file_operation_frame, 
            text=f"{self.icons['clear']} 清空内容", 
            command=self.clear_file_fields,
            style="TButton",
            width=12
        )
        self.clear_file_button.pack(pady=5, padx=2, anchor=tk.CENTER)
        
        # 添加退出按钮到文件操作区域
        self.exit_button_file = ttk.Button(
            self.file_operation_frame, 
            text=f"{self.icons['exit']} 退出程序", 
            command=self.exit_program,
            style="Exit.TButton",
            width=12
        )
        self.exit_button_file.pack(pady=5, padx=2, anchor=tk.CENTER)
        
        # 创建文本处理区域
        self.text_content_frame = ttk.Frame(self.content_frame)
        
        # 创建输入文本区域
        input_frame = ttk.LabelFrame(self.text_content_frame, text="输入藏文文本", padding="15")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=10,
            font=self.tibetan_font,
            bg=self.secondary_color,
            fg=self.text_color,
            insertbackground=self.primary_color,
            selectbackground=self.primary_color,
            selectforeground="white",
            padx=10,
            pady=10,
            borderwidth=1,
            relief="solid"
        )
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建输出文本区域
        output_frame = ttk.LabelFrame(self.text_content_frame, text="分词结果", padding="15")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=10,
            font=self.tibetan_font,
            bg=self.secondary_color,
            fg=self.text_color,
            padx=10,
            pady=10,
            borderwidth=1,
            relief="solid"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建文件处理区域
        self.file_content_frame = ttk.Frame(self.content_frame)
        
        # 创建文件选择框架
        file_select_frame = ttk.LabelFrame(self.file_content_frame, text="输入文件", padding="15")
        file_select_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # 创建文件选择区域，包含输入框和浏览按钮
        file_input_frame = ttk.Frame(file_select_frame)
        file_input_frame.pack(fill=tk.X, padx=5, pady=8)
        
        # 输入文件
        self.input_file_var = tk.StringVar()
        ttk.Entry(
            file_input_frame, 
            textvariable=self.input_file_var, 
            width=50,
            font=self.text_font
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # 添加浏览按钮到文件选择区域
        ttk.Button(
            file_input_frame, 
            text=f"{self.icons['browse']} 浏览", 
            command=self.browse_input_file,
            style="TButton",
            width=8
        ).pack(side=tk.RIGHT, padx=2)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(self.file_content_frame, text="处理结果预览", padding="15")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=10,
            font=self.tibetan_font,  # 使用藏文字体
            bg=self.secondary_color,
            fg=self.text_color,
            padx=10,
            pady=10,
            borderwidth=1,
            relief="solid"
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 状态栏和进度条
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            bottom_frame, 
            variable=self.progress_var, 
            maximum=100,
            style="Horizontal.TProgressbar",
            length=100
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # 设置进度条样式
        self.style.configure("Horizontal.TProgressbar", 
                           background=self.accent_color,
                           troughcolor=self.secondary_color,
                           borderwidth=0,
                           thickness=15)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(
            bottom_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            font=self.text_font,
            background=self.primary_color,
            foreground="white",
            padding=5
        )
        self.status_bar.pack(fill=tk.X)
        
        # 默认显示文本处理模式
        self.show_text_mode()
        
        # 检查模型是否加载成功
        if not os.path.exists(self.tokenizer.model_path) or not os.path.exists(self.tokenizer.datas_pkl):
            messagebox.showerror("错误", "模型文件或数据文件不存在，请确保以下文件存在：\n"
                                 f"1. {self.tokenizer.model_path}\n"
                                 f"2. {self.tokenizer.datas_pkl}")
    
    def show_text_mode(self):
        """显示文本处理模式"""
        # 隐藏文件处理区域
        self.file_content_frame.pack_forget()
        self.file_operation_frame.pack_forget()
        
        # 显示文本处理区域
        self.text_content_frame.pack(fill=tk.BOTH, expand=True)
        self.text_operation_frame.pack(fill=tk.BOTH, expand=True)
        
        # 更新按钮状态
        self.text_mode_button.configure(style="TextMode.TButton")
        self.file_mode_button.configure(style="FileMode.TButton")
        
        self.status_var.set("文本处理模式")
    
    def show_file_mode(self):
        """显示文件处理模式"""
        # 隐藏文本处理区域
        self.text_content_frame.pack_forget()
        self.text_operation_frame.pack_forget()
        
        # 显示文件处理区域
        self.file_content_frame.pack(fill=tk.BOTH, expand=True)
        self.file_operation_frame.pack(fill=tk.BOTH, expand=True)
        
        # 更新按钮状态
        self.file_mode_button.configure(style="TextMode.TButton")
        self.text_mode_button.configure(style="FileMode.TButton")
        
        self.status_var.set("文件处理模式")

    def process_text(self):
        """处理文本框中的文本"""
        input_text = self.input_text.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showinfo("提示", "请输入要处理的藏文文本")
            return
        
        self.status_var.set("处理中...")
        self.process_text_button.config(state=tk.DISABLED)
        self.progress_var.set(10)
        
        # 使用线程处理文本，避免界面卡顿
        def process_thread():
            start_time = time.time()
            
            # 更新进度条
            self.root.after(0, lambda: self.progress_var.set(30))
            
            result = self.tokenizer.process_text(input_text)
            
            # 更新进度条
            self.root.after(0, lambda: self.progress_var.set(90))
            
            end_time = time.time()
            
            # 更新界面
            self.root.after(0, lambda: self.update_output_text(result, end_time - start_time))
        
        threading.Thread(target=process_thread).start()

    def update_output_text(self, result, process_time):
        """更新输出文本框"""
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", result)
        self.status_var.set(f"处理完成，用时: {process_time:.2f} 秒")
        self.process_text_button.config(state=tk.NORMAL)
        self.progress_var.set(100)

    def clear_text(self):
        """清空文本处理标签页的文本框"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.status_var.set("就绪")
        self.progress_var.set(0)
        
    def copy_result(self):
        """复制分词结果到剪贴板"""
        result_text = self.output_text.get("1.0", tk.END).strip()
        if not result_text:
            messagebox.showinfo("提示", "没有可复制的结果")
            return
            
        self.root.clipboard_clear()
        self.root.clipboard_append(result_text)
        self.status_var.set("结果已复制到剪贴板")

    def browse_input_file(self):
        """浏览输入文件"""
        filename = filedialog.askopenfilename(
            title="选择输入文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if filename:
            self.input_file_var.set(filename)
            self.status_var.set(f"已选择文件: {os.path.basename(filename)}")

    def process_file(self):
        """处理文件"""
        input_file = self.input_file_var.get()
        
        if not input_file:
            messagebox.showinfo("提示", "请选择输入文件")
            return
        
        if not os.path.exists(input_file):
            messagebox.showerror("错误", f"输入文件不存在: {input_file}")
            return
        
        self.status_var.set("处理文件中...")
        self.process_file_button.config(state=tk.DISABLED)
        self.progress_var.set(10)
        
        # 使用线程处理文件，避免界面卡顿
        def process_file_thread():
            start_time = time.time()
            
            try:
                # 更新进度条
                self.root.after(0, lambda: self.progress_var.set(30))
                
                # 处理文本
                result = self.tokenizer.process_file(input_file)
                
                # 更新进度条
                self.root.after(0, lambda: self.progress_var.set(90))
                
                end_time = time.time()
                
                # 更新界面
                self.root.after(0, lambda: self.update_file_result(True, end_time - start_time, input_file, result))
            
            except Exception as e:
                # 处理错误
                self.root.after(0, lambda: self.update_file_result(False, 0, str(e), None))
        
        threading.Thread(target=process_file_thread).start()

    def save_result_file(self):
        """保存处理结果到文件"""
        result_content = self.result_text.get("1.0", tk.END)
        if not result_content.strip():
            messagebox.showinfo("提示", "没有可保存的处理结果")
            return
        
        # 获取默认文件名
        input_file = self.input_file_var.get()
        if input_file:
            default_name = os.path.splitext(input_file)[0] + "_分词结果.txt"
        else:
            default_name = "分词结果.txt"
        
        # 打开保存对话框
        filename = filedialog.asksaveasfilename(
            title="保存分词结果",
            defaultextension=".txt",
            initialfile=os.path.basename(default_name),
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(result_content)
                messagebox.showinfo("成功", f"结果已保存到:\n{filename}")
                self.status_var.set(f"结果已保存到: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出错:\n{str(e)}")

    def update_file_result(self, success, process_time, message, result_content):
        """更新文件处理结果"""
        self.progress_var.set(100)
        self.process_file_button.config(state=tk.NORMAL)
        
        if success:
            self.status_var.set(f"文件处理完成，用时: {process_time:.2f} 秒")
            self.result_text.delete("1.0", tk.END)
            
            # 在结果预览框中显示处理后的全部内容
            if result_content:
                self.result_text.insert("1.0", result_content)
            else:
                self.result_text.insert("1.0", f"处理成功！\n\n输入文件: {message}\n\n处理用时: {process_time:.2f} 秒")
        else:
            self.status_var.set("处理失败")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", f"处理失败！\n\n错误信息: {message}")
            messagebox.showerror("错误", f"处理文件时出错:\n{message}")

    def clear_file_fields(self):
        """清空文件处理标签页的字段"""
        self.input_file_var.set("")
        self.result_text.delete("1.0", tk.END)
        self.progress_var.set(0)
        self.status_var.set("就绪")

    def clear_all(self):
        """清空所有内容"""
        self.clear_text()
        self.clear_file_fields()
        self.status_var.set("已清空所有内容")
        self.progress_var.set(0)

    def exit_program(self):
        """退出程序"""
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    # 设置窗口图标
    try:
        # 尝试设置窗口图标，如果图标文件不存在则忽略
        root.iconbitmap("icon.ico")
    except:
        pass
    app = TibetanTokenizerGUI(root)
    root.mainloop()
