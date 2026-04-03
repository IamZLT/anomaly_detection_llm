#!/usr/bin/env python3
"""
启动Flask Web应用的入口脚本
在项目根目录运行此脚本即可启动Web界面

使用方法:
    python run_app.py
"""

import os
import sys

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(project_root, 'app')

# 添加app目录到Python路径
sys.path.insert(0, app_dir)

# 导入Flask应用
from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Qwen3-VL 微调模型 Web 应用")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print(f"应用目录: {app_dir}")
    print("=" * 60)
    print("\n访问 http://localhost:5000 使用Web界面")
    print("按 Ctrl+C 停止服务器\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
