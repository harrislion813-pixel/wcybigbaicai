#!/usr/bin/env python3
"""
批量叶色表型提取脚本。

Usage:
    python batch_extract.py --config ../config.yaml
    python batch_extract.py --input data/raw_images/ --output results/phenotypes.csv
    python batch_extract.py --input data/raw_images/ --method unet --model models/unet.pth
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd
from src.pipeline import LeafColorPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="大白菜叶色表型批量提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件
  python batch_extract.py --config ../config.yaml

  # 快速命令行模式 (ExG分割)
  python batch_extract.py --input ./data/raw/ --output ./output/phenotypes.csv

  # 深度学习模式 (U-Net分割)
  python batch_extract.py --input ./data/raw/ --method unet --model ./models/unet.pth

  # 不汇总样本 (保留每张图像的独立记录)
  python batch_extract.py --input ./data/raw/ --no-aggregate
        """
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML配置文件路径")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="输入图像目录")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出CSV文件路径")
    parser.add_argument("--method", "-m", type=str,
                        choices=["exg", "grabcut", "unet", "sam", "auto"],
                        default="exg",
                        help="叶片分割方法 (default: exg)")
    parser.add_argument("--model", type=str, default=None,
                        help="U-Net/SAM模型权重路径")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="计算设备 (default: cpu)")
    parser.add_argument("--white-balance", "-wb", type=str,
                        default="gray_world",
                        choices=["gray_world", "perfect_reflector", "none"],
                        help="白平衡方法 (default: gray_world)")
    parser.add_argument("--id-pattern", type=str, default=None,
                        help="样本ID正则提取模式 (e.g. '(\\\\w+)_rep')")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="不按样本ID汇总, 保留每张图像的独立记录")
    parser.add_argument("--visualize", action="store_true",
                        help="为每张图像生成可视化结果并保存")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细进度")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 加载配置 ----
    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {args.config}")
    else:
        # 默认配置
        config = {
            "segmentation": {
                "method": args.method,
            },
            "output": {
                "format": "csv",
                "separate_visualization": args.visualize,
            },
        }
        if args.model:
            config["segmentation"]["model_path"] = args.model
            config["segmentation"]["device"] = args.device

    # ---- 命令行参数覆盖 ----
    if args.method:
        config.setdefault("segmentation", {})["method"] = args.method
    if args.model:
        config.setdefault("segmentation", {})["model_path"] = args.model
        config["segmentation"]["device"] = args.device

    # ---- 输入输出 ----
    input_dir = args.input or config.get("input", {}).get("image_dir", "./data/raw_images/")
    output_csv = args.output

    if not output_csv:
        output_dir = config.get("input", {}).get("output_dir", "./output/")
        output_csv = str(Path(output_dir) / "leaf_color_phenotypes.csv")

    # ---- 运行流水线 ----
    print("=" * 60)
    print("大白菜叶色表型批量提取")
    print("=" * 60)
    print(f"  Input:        {input_dir}")
    print(f"  Output:       {output_csv}")
    print(f"  Segmentation: {args.method}")
    print(f"  White balance: {args.white_balance}")
    print(f"  Aggregate:    {not args.no_aggregate}")
    print("=" * 60)

    pipeline = LeafColorPipeline(config)
    df = pipeline.process_batch(
        image_dir=input_dir,
        output_csv=output_csv,
        id_pattern=args.id_pattern,
        group_by_sample=not args.no_aggregate,
        white_balance=args.white_balance,
        save_visualizations=args.visualize,
        verbose=args.verbose,
    )

    if df.empty:
        print("\n[WARNING] No phenotypes extracted. Check your input directory.")
        return 1

    # ---- 输出摘要 ----
    print("\n" + "=" * 60)
    print("提取结果摘要")
    print("=" * 60)
    print(f"  样本数:      {df.shape[0]}")
    print(f"  性状数:      {df.shape[1]}")
    print(f"  总特征维度:  {df.shape[0] * df.shape[1]:,}")

    # 关键性状摘要
    key_traits = [
        "CIELAB_L_mean", "CIELAB_A_mean", "CIELAB_B_mean",
        "CIELAB_greenness", "GLI", "DGCI", "VARI",
        "HSV_H_mean", "Uniformity_dE_mean"
    ]
    existing_traits = [t for t in key_traits if t in df.columns]
    if existing_traits:
        print("\n  关键性状统计:")
        print(df[existing_traits].describe().to_string())

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
