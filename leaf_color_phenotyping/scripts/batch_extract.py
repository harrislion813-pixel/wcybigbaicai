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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


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
    parser.add_argument(
        "--config", "-c", type=str, default=str(DEFAULT_CONFIG_PATH),
        help=f"YAML配置文件路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="输入图像目录")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径（CSV/JSON/Excel）")
    parser.add_argument("--method", "-m", type=str,
                        choices=["exg", "grabcut", "unet", "sam", "auto"],
                        default=None,
                        help="叶片分割方法（未指定时读取配置，默认 exg）")
    parser.add_argument("--model", type=str, default=None,
                        help="U-Net/SAM模型权重路径")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"],
                        help="计算设备（未指定时读取配置，默认 cpu）")
    parser.add_argument("--white-balance", "-wb", type=str,
                        default=None,
                        choices=["gray_world", "perfect_reflector", "gray_card", "none"],
                        help="白平衡方法（未指定时读取配置，默认 none）")
    parser.add_argument("--id-pattern", type=str, default=None,
                        help="样本ID正则提取模式 (e.g. '(\\\\w+)_rep')")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="不按样本ID汇总, 保留每张图像的独立记录")
    parser.add_argument("--visualize", action="store_true",
                        help="为每张图像生成可视化结果并保存")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细进度")
    parser.add_argument("--allow-partial", action="store_true",
                        help="即使部分图像处理失败，也以成功状态退出")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 加载配置 ----
    config = {}
    if args.config:
        config_path = Path(args.config).resolve()
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        config["_config_dir"] = str(config_path.parent)
        print(f"Loaded config from: {args.config}")
    else:
        # 默认配置
        config = {
            "segmentation": {
                "method": args.method or "exg",
                "device": args.device or "cpu",
            },
            "output": {
                "format": "csv",
                "separate_visualization": args.visualize,
            },
        }
        if args.model:
            config["segmentation"]["model_path"] = str(Path(args.model).resolve())

    # ---- 命令行参数覆盖 ----
    if args.method:
        config.setdefault("segmentation", {})["method"] = args.method
    if args.model:
        config.setdefault("segmentation", {})["model_path"] = str(Path(args.model).resolve())
    if args.device:
        config.setdefault("segmentation", {})["device"] = args.device

    # ---- 输入输出 ----
    config_dir = Path(config.get("_config_dir", "."))
    configured_input = config.get("input", {}).get("image_dir", "./data/raw_images/")
    if args.input:
        input_dir = args.input
    else:
        input_path = Path(configured_input)
        input_dir = str(input_path if input_path.is_absolute() else config_dir / input_path)
    output_csv = args.output

    if not output_csv:
        output_dir = config.get("input", {}).get("output_dir", "./output/")
        output_dir_path = Path(output_dir)
        if not output_dir_path.is_absolute():
            output_dir_path = config_dir / output_dir_path
        output_cfg = config.get("output", {})
        output_format = str(output_cfg.get("format", "csv")).lower()
        extension = {"csv": ".csv", "excel": ".xlsx", "json": ".json"}.get(output_format)
        if extension is None:
            raise ValueError("output.format must be one of: csv, excel, json")
        table_name = output_cfg.get("phenotype_table_name", "leaf_color_phenotypes")
        output_csv = str(output_dir_path / f"{table_name}{extension}")

    effective_method = config.get("segmentation", {}).get("method", "exg")
    effective_white_balance = (
        args.white_balance or config.get("imaging", {}).get("white_balance", "none")
    )
    gray_roi = config.get("imaging", {}).get("gray_card_rgb")
    save_visualizations = (
        args.visualize or config.get("output", {}).get("separate_visualization", False)
    )

    # ---- 运行流水线 ----
    print("=" * 60)
    print("大白菜叶色表型批量提取")
    print("=" * 60)
    print(f"  Input:        {input_dir}")
    print(f"  Output:       {output_csv}")
    print(f"  Segmentation: {effective_method}")
    print(f"  White balance: {effective_white_balance}")
    print(f"  Aggregate:    {not args.no_aggregate}")
    print("=" * 60)

    pipeline = LeafColorPipeline(config)
    df = pipeline.process_batch(
        image_dir=input_dir,
        output_csv=output_csv,
        id_pattern=args.id_pattern,
        group_by_sample=not args.no_aggregate,
        white_balance=effective_white_balance,
        gray_roi=gray_roi,
        save_visualizations=save_visualizations,
        verbose=args.verbose,
    )

    if df.empty:
        print("\n[WARNING] No phenotypes extracted. Check your input directory.")
        return 1

    if pipeline.last_batch_failures:
        print(f"\n[ERROR] {len(pipeline.last_batch_failures)} image(s) failed.")
        if not args.allow_partial:
            print("Use --allow-partial to accept an incomplete phenotype table.")
            return 2

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
