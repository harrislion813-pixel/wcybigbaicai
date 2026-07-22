# 🍃 大白菜叶色表型提取工具包

基于图像的 **大白菜叶色性状高通量提取** 工具包，专为 **GWAS 关联分析** 设计。

## 核心能力

| 模块 | 功能 | 提取特征数 |
|------|------|-----------|
| **预处理** | RAW读取、白平衡、ColorChecker颜色校准 | — |
| **叶片分割** | ExG阈值 / GrabCut / U-Net / SAM | — |
| **颜色特征** | RGB、HSV、CIELAB、YCbCr + 颜色矩 + 直方图 | ~150 |
| **植被指数** | VARI, GLI, ExG, DGCI, CIVE, MGRVI 等 13 个指数 | ~39 |
| **色度坐标** | CIE xyY, CIE u'v' | ~8 |
| **纹理特征** | GLCM (对比度/同质性/能量/相关性/ASM) | ~60 |
| **形状特征** | 面积/周长/圆度/偏心率/坚实度/长宽比 | ~10 |
| **均匀性** | CIELAB空间色差分布 (斑驳/黄化不均匀性) | ~8 |
| **合计** | | **~275 个性状** |

## 快速开始

### 安装

```bash
cd leaf_color_phenotyping

# 基础安装 (仅传统分割)
pip install numpy opencv-python scikit-image pandas PyYAML

# 完整安装 (含深度学习)
pip install -r requirements.txt

# 开发与测试
pip install -r requirements-dev.txt
pytest
```

### 单张图像快速验证

```python
from src.pipeline import LeafColorPipeline

pipeline = LeafColorPipeline()
result = pipeline.process_single(
    "data/raw_images/BJC-001_rep1.jpg",
    sample_id="BJC-001",
    replicate="rep1",
    developmental_stage="heading",
    return_visualization=True,
)

# 查看提取的特征
for k, v in result["features"].items():
    print(f"  {k}: {v:.4f}")

# 保存可视化
import cv2
from src.utils import write_image_rgb
write_image_rgb("output/BJC-001_vis.jpg", result["visualization"])
```

### 批量处理 (GWAS表型数据生成)

```bash
# 命令行
python scripts/batch_extract.py \
    --input data/raw_images/ \
    --output results/phenotypes.csv \
    --method exg \
    --verbose

# 或 Python
python -c "
from src.pipeline import LeafColorPipeline
pipeline = LeafColorPipeline()
df = pipeline.process_batch('data/raw_images/', output_csv='phenotypes.csv')
print(df.describe())
"
```

## 输出表型表格式

输出 CSV 示例（用于 GWAS 分析的输入）：

| sample_id | n_replicates | CIELAB_L_mean | CIELAB_A_mean | CIELAB_B_mean | GLI | DGCI | ... |
|-----------|-------------|---------------|---------------|---------------|-----|------|-----|
| BJC-001 | 3 | 45.23 | -18.56 | 32.10 | 0.234 | 0.567 | ... |
| BJC-002 | 3 | 38.91 | -15.23 | 28.44 | 0.198 | 0.612 | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

每个性状的 `_mean`、`_std`、`_cv` 列分别代表：
- 原始特征名中的 `_mean` / `_std` 表示**单张叶片 ROI 内像素**的统计量，
  例如 `CIELAB_L_mean` 和 `CIELAB_L_std`。
- 多重复汇总后的均值保留原始特征名，例如 `CIELAB_L_mean`。
- 重复间标准差和变异系数追加 `_rep_std` / `_rep_cv`，例如
  `CIELAB_L_mean_rep_std` 和 `CIELAB_L_mean_rep_cv`。
- 当重复均值接近零时，CV 输出为缺失值，避免产生无意义的极大数。

## 项目结构

```
leaf_color_phenotyping/
├── config.yaml                  # 全局配置文件
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── utils.py                 # 工具函数、色彩空间转换、色差公式
│   ├── preprocessing.py         # RAW读取、白平衡、颜色校准(CCM)
│   ├── segmentation.py          # 叶片分割 (ExG/GrabCut/U-Net/SAM)
│   ├── color_features.py        # 多颜色空间特征提取
│   ├── vegetation_indices.py    # RGB植被指数 + SPAD估算
│   ├── texture_features.py      # GLCM纹理 + 形状特征 + 均匀性
│   └── pipeline.py              # 完整流水线编排器
├── scripts/
│   ├── batch_extract.py         # 批量提取命令行工具
│   └── train_segmentation.py    # U-Net分割模型训练脚本
├── notebooks/                   # Jupyter Notebook (交互式分析)
├── models/                      # 预训练模型存放目录
├── data/                        # 示例数据
└── output/                      # 结果输出目录
```

## 分割方法选择指南

| 方法 | 速度 | 精度 | GPU需求 | 适用场景 |
|------|------|------|---------|---------|
| `exg` | ⚡⚡⚡ | ⭐⭐ | 不需要 | 绿色叶片 + 均匀背景 |
| `grabcut` | ⚡⚡ | ⭐⭐⭐ | 不需要 | 复杂背景, 无GPU |
| `unet` | ⚡⚡ | ⭐⭐⭐⭐⭐ | 需要 | 大规模批处理 (需先训练) |
| `sam` | ⚡ | ⭐⭐⭐⭐⭐ | 需要 | 零样本, 无需标注 |

## 颜色校准流程 (推荐)

为确保不同批次图像颜色可比，正式实验应进行颜色校准。流水线不会再把
“启用但没有 CCM”的配置当作成功：应先计算并保存校正矩阵，再在配置中启用。

```python
from src.preprocessing import ImagePreprocessor

# 1. 拍摄含ColorChecker色卡的参考图像
# 2. 用色卡工具或人工标注提取24个色块的RGB均值，保存为 (24,3) 数组
import numpy as np
measured_rgb = np.load("data/colorchecker/measured_rgb.npy")

# 3. 计算颜色校正矩阵
preprocessor = ImagePreprocessor(calibration_method="polynomial", polynomial_degree=2)
ccm = preprocessor.compute_color_correction_matrix(measured_rgb)

# 保存 CCM，供批处理配置复用
np.save("models/colorchecker_ccm.npy", ccm)

# 4. 对后续所有图像应用相同的CCM
corrected = preprocessor.apply_color_correction(some_image)
```

批处理配置：

```yaml
color_calibration:
  enabled: true
  method: polynomial
  polynomial_degree: 2
  ccm_file: models/colorchecker_ccm.npy
```

`ccm_file` 相对于配置文件所在目录解析。若不需要跨批次比较，可保持
`enabled: false`，但输出应明确记录这一实验条件。

## 批处理失败策略

批处理会继续检查剩余图像，同时在输出目录写入 `*_failures.csv`。只要存在
失败图像，命令行默认返回非零状态，防止不完整表型表被误当作完整结果；只有
明确接受部分结果时才使用 `--allow-partial`。

## GWAS分析衔接

生成的表型表可直接对接 GWAS 分析:

```r
# R 代码示例 — GWAS分析衔接
pheno <- read.csv("phenotypes.csv", row.names = 1)

# Step 1: 计算BLUP (消除环境/重复效应)
library(lme4)
blup_model <- lmer(CIELAB_A_mean ~ (1|sample_id) + developmental_stage, data = pheno)
blups <- ranef(blup_model)$sample_id

# Step 2: 计算广义遗传力
# H² = Vg / (Vg + Ve/n)

# Step 3: 输入GAPIT3
# myGAPIT <- GAPIT(Y = blups, G = geno, PCA.total = 3, model = "FarmCPU")
```

## 引用

如果使用本工具包, 请引用相关方法文献 (详见各模块文档字符串中的文献引用)。

## License

MIT License
