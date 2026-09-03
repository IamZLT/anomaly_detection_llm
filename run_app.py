#!/usr/bin/env python3
"""
启动 Flask Web 应用。

用法:
  python run_app.py --model-path /path/to/outputs/train/.../grpo_final
  python run_app.py --config configs/qwen35_9b_zeroshot.yaml
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(project_root, "app")
sys.path.insert(0, project_root)
sys.path.insert(0, app_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-VL Web 推理")
    default_cfg = os.path.join(project_root, "configs", "qwen35_9b_zeroshot.yaml")
    parser.add_argument("--config", type=str, default=default_cfg, help="YAML 配置")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="覆盖 yaml 中的 inference.model_path（基座或 LoRA 输出目录）",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()

    os.environ["QWEN_WEB_CONFIG"] = os.path.abspath(args.config)
    if args.model_path:
        os.environ["QWEN_WEB_MODEL_PATH"] = os.path.abspath(args.model_path)

    from app import app

    print("=" * 60)
    print("Qwen-VL Web 应用")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print(f"配置文件: {os.environ['QWEN_WEB_CONFIG']}")
    if args.model_path:
        print(f"模型路径: {os.environ['QWEN_WEB_MODEL_PATH']}")
    print("=" * 60)
    print(f"\n访问 http://localhost:{args.port} 使用 Web 界面")
    print("按 Ctrl+C 停止服务器\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
