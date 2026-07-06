"""
leaf_color_phenotyping — 基于图像表型的大白菜叶色性状提取工具包

核心模块:
    preprocessing  — RAW读取、白平衡、颜色校准
    segmentation  — 叶片分割 (ExG / GrabCut / U-Net / SAM)
    color_features — 多颜色空间特征提取
    vegetation_indices — RGB植被指数
    texture_features — GLCM纹理 + 叶片形状
    pipeline — 完整流水线编排
    utils — 工具函数

Usage:
    from leaf_color_phenotyping.src.pipeline import LeafColorPipeline

    pipeline = LeafColorPipeline()
    df = pipeline.process_batch("data/raw_images/", output_csv="phenotypes.csv")
"""

__version__ = "1.0.0"
__author__ = "Leaf Color Phenotyping Team"
