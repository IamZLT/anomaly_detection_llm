# Flask Web应用 - Qwen3-VL微调模型推理界面

## 功能特性

- 📷 上传图像（支持拖拽）
- 💬 自定义提示词
- 🤖 使用微调模型进行推理
- 📍 自动检测并标注bbox边界框
- 🎨 美观的Web界面

## 目录结构

```
app/
├── app.py              # Flask应用主文件
├── templates/          # HTML模板
│   └── index.html     # 主页面
├── static/            # 静态文件
│   └── outputs/       # 标注后的图像输出目录
├── uploads/          # 用户上传的图像目录
└── README.md          # 本文件
```

## 使用方法

### 1. 安装依赖

```bash
pip install flask transformers torch pillow
```

### 2. 配置模型路径

编辑 `app.py`，修改 `MODEL_PATH` 变量为你的模型路径：

```python
MODEL_PATH = os.path.join(os.path.dirname(APP_DIR), "outputs/grounding_mvtec_YYYYMMDD_HHMMSS/final_model")
```

### 3. 启动应用

**推荐方式（在项目根目录）：**

```bash
cd /data2/zlt/anomaly_detection_llm
python run_app.py
```

或者直接在app目录：

```bash
cd /data2/zlt/anomaly_detection_llm/app
python app.py
```

### 4. 访问Web界面

在浏览器中打开：`http://localhost:5000`

## 使用说明

1. **上传图像**：点击上传区域或拖拽图像文件
2. **输入提示词**：在文本框中输入你的问题或提示词
3. **开始推理**：点击"开始推理"按钮
4. **查看结果**：
   - 查看模型回复
   - 如果检测到bbox，会在右侧显示标注后的图像
   - 查看bbox坐标信息

## 配置参数

可以在 `app.py` 中修改以下参数：

- `MODEL_PATH`: 模型路径
- `MAX_IMAGE_SIZE`: 图像最大尺寸（默认512）
- `FACTOR`: 图像尺寸因子（默认28）
- `MAX_NEW_TOKENS`: 最大生成token数（默认512）
- `TEMPERATURE`: 生成温度（默认0.7）
- `TOP_P`: Top-p采样（默认0.9）

## 注意事项

- 首次加载模型可能需要几分钟时间
- 模型会缓存在内存中，后续推理会更快
- 上传的图像最大为16MB
- 标注后的图像会保存在 `static/outputs/` 目录
