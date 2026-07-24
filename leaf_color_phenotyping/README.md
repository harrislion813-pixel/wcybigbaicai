# 大白菜叶色表型提取：从零开始操作指南

这是一套从叶片图片中批量提取颜色、植被指数和纹理特征的 Python 工具。

如果你第一次使用本项目，按本文的顺序操作即可。最稳妥的入门方案是：

1. 把图片放进 data/raw_images。
2. 使用传统 ExG 分割，不需要模型，也不需要显卡。
3. 输出 CSV 文件并检查分割图。
4. 确认结果可靠后，再考虑 U-Net、SAM 或颜色校准。

> 重要：本工具可以自动计算表型，但不能替代实验设计和人工质控。正式分析前，请抽查分割结果，并结合 QC 列过滤失败样本。

---

## 一、最快跑通：只做这 5 步

以下命令都要在 leaf_color_phenotyping 项目目录中执行。

### 第 1 步：打开终端并进入项目目录

Windows PowerShell 示例：

~~~powershell
cd C:\你的项目路径\wcybigbaicai\leaf_color_phenotyping
~~~

macOS 或 Linux 示例：

~~~bash
cd /你的项目路径/wcybigbaicai/leaf_color_phenotyping
~~~

### 第 2 步：创建并启用虚拟环境

Windows PowerShell：

~~~powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
~~~

macOS 或 Linux：

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

启用成功后，命令行前面通常会出现 (.venv)。

如果 Windows 上没有 python 命令，可以把本文命令中的 python 换成 py。

### 第 3 步：安装最基本的依赖

只使用 ExG、GrabCut 和自动传统分割时，安装下面这些包即可：

~~~bash
python -m pip install --upgrade pip
python -m pip install numpy opencv-python scikit-image pandas PyYAML openpyxl
~~~

如果要读取相机 RAW 文件，再安装：

~~~bash
python -m pip install rawpy
~~~

如果要训练或运行 U-Net，请安装完整依赖：

~~~bash
python -m pip install -r requirements.txt
~~~

完整依赖包含 PyTorch、分割模型和绘图工具，下载体积较大。第一次试用本项目时，不必先安装完整依赖。

### 第 4 步：放入图片

在项目目录下建立或使用下面的文件夹：

~~~text
leaf_color_phenotyping/
└── data/
    └── raw_images/
        ├── BJC-001_rep1.jpg
        ├── BJC-001_rep2.jpg
        ├── BJC-002_rep1.jpg
        └── BJC-002_rep2.jpg
~~~

支持的常见图片包括 JPG、JPEG、PNG、TIF、TIFF 和 BMP。安装 rawpy 后，还可读取 RAF、DNG、CR2、NEF、ARW 等常见 RAW 格式。RAW 默认转换为带相机白平衡的标准 sRGB，并关闭逐图自动增亮，以便与下游颜色空间转换保持一致。

建议：

- 一张图片尽量只放一片主要叶片。
- 叶片不要紧贴图片边缘。
- 背景与绿色叶片的颜色差异越大，自动分割越稳定。
- 同一批图片尽量使用相同光源、相机、曝光和拍摄距离。
- 不要先用社交软件压缩图片。

### 第 5 步：复制命令并运行

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/leaf_color_phenotypes.csv --method exg --white-balance gray_world --visualize --verbose
~~~

运行结束后，重点查看：

~~~text
output/
├── leaf_color_phenotypes.csv
└── visualizations/
    ├── BJC-001_rep1_vis.jpg
    └── ...
~~~

如果 CSV 已生成，而且 visualizations 中的绿色轮廓基本覆盖叶片、没有大面积覆盖背景，说明最基本流程已经跑通。

---

## 二、功能实现模块：图片是怎样变成表型数据的

可以把整个程序想象成一条“叶片图片加工流水线”：

~~~text
原始图片 + config.yaml
        ↓
1. 读取图片并统一颜色
        ↓
2. 从背景中抠出叶片
        ↓
3. 只在叶片区域计算特征
        ↓
4. 生成质量检查指标
        ↓
5. 合并同一样本的重复图片
        ↓
CSV / Excel / JSON + 分割检查图
~~~

批处理入口 scripts/batch_extract.py 像“总开关”：它读取命令和配置文件，然后让 src/pipeline.py 按顺序调用下面的功能模块。

### 模块 1：读取和校正图片

对应文件：src/preprocessing.py、src/utils.py

这个模块负责把不同来源的图片整理成程序能够统一处理的 RGB 数据。

它主要做四件事：

1. 读取 JPG、PNG、TIFF 或 RAW 图片。
2. 把像素统一换算到 0 到 1。
3. 根据设置执行灰度世界、灰卡等白平衡。
4. 如果启用了 CCM，再进行颜色校准。

可以把它理解为：先把不同相机拍出的照片“调整到同一把尺子上”，再比较叶片颜色。

如果只想快速使用，采用 gray_world，并关闭颜色校准即可。

### 模块 2：从背景中找到叶片

对应文件：src/segmentation.py

分割就是给叶片“抠图”。程序会生成一张只有黑色和白色的掩膜：

~~~text
白色区域 = 叶片，参与特征计算
黑色区域 = 背景，不参与特征计算
~~~

ExG 根据“叶片通常比背景更绿”来识别叶片；GrabCut 会结合颜色和画面位置进一步寻找前景；U-Net 和 SAM 则使用模型预测叶片范围。

这是整条流程中最需要检查的一步。如果把背景误认为叶片，后面的颜色、植被指数和纹理都会一起受到影响。因此第一次处理新批次图片时，务必添加 --visualize。

### 模块 3：计算叶片特征

只有掩膜中的白色叶片区域会进入特征计算。

| 特征类型 | 对应文件 | 通俗解释 |
|---|---|---|
| 颜色特征 | src/color_features.py | 叶片有多亮、多绿、多黄，以及颜色分布是否均匀 |
| 植被指数 | src/vegetation_indices.py | 用 R、G、B 通道组合出 ExG、VARI、DGCI 等指标 |
| 纹理特征 | src/texture_features.py | 叶面看起来是否平滑、粗糙、均匀或具有方向性 |
| 形状特征 | src/texture_features.py | 叶片面积、周长、圆度、长宽和紧实程度 |

例如，CIELAB_L_mean 表示叶片区域的平均明亮程度；ExG 表示叶片区域的平均超绿指数，ExG_std 表示单张图片内叶片像素超绿指数的标准差。

### 模块 4：给结果附上质量检查指标

对应文件：src/pipeline.py

程序不会只给出表型值，还会同时记录：

- 识别出的叶片有多少像素。
- 叶片占整张图片的比例。
- 是否完全没有识别出叶片。
- 叶片外接框的位置和大小。

这些列以 QC_ 开头。它们像体检报告中的“异常提示”，帮助你找到需要重新分割或剔除的图片。

程序不会擅自删除异常样本，因为不同拍摄距离下，合理的叶片面积比例可能不同。应先查看分割图，再根据自己的实验条件设置筛选标准。

### 模块 5：识别样本并合并重复图片

对应文件：src/utils.py、src/pipeline.py

程序先从文件名得到 sample_id，再把同一样本的重复图片放在一起。例如：

~~~text
BJC-001_rep1.jpg ┐
BJC-001_rep2.jpg ├─→ sample_id = BJC-001
BJC-001_rep3.jpg ┘
~~~

假设三张图片都计算出了 CIELAB_L_mean，聚合后会得到：

~~~text
CIELAB_L_mean         三张重复图片的平均值
CIELAB_L_mean_rep_std 三张重复图片之间的标准差
CIELAB_L_mean_rep_cv  三张重复图片之间的变异系数
n_replicates          实际参与计算的重复图片数
~~~

因此，rep_std 或 rep_cv 很大时，通常意味着重复图片差异较大，应返回原图和分割图检查。

### 模块 6：保存表格和检查图

对应文件：src/pipeline.py

最后，程序根据输出文件扩展名保存 CSV、XLSX 或 JSON，并在启用 --visualize 时生成分割检查图。

如果有图片处理失败，还会单独保存以 _failures.csv 结尾的报告。这样成功结果不会丢失，失败原因也能被追踪。

### 用一个样本串起整个过程

以 BJC-001_rep1.jpg 为例：

1. 程序读取图片，执行 gray_world 白平衡。
2. ExG 生成叶片掩膜，背景被排除。
3. 在叶片像素中计算 CIELAB、ExG、纹理和形状。
4. 记录叶片面积比例等 QC 指标。
5. 与 BJC-001 的其他重复图片合并。
6. 在结果表中生成 sample_id 为 BJC-001 的一行数据。

如果使用 --no-aggregate，第 5 步会被跳过，每张图片各保留一行。

### 想增加新功能时改哪里

| 想增加的功能 | 优先修改的位置 |
|---|---|
| 新植被指数 | src/vegetation_indices.py |
| 新颜色统计量 | src/color_features.py |
| 新纹理或形状指标 | src/texture_features.py |
| 新分割方法 | src/segmentation.py，并在 create_segmenter 中注册 |
| 新输出列或聚合规则 | src/pipeline.py |
| 新命令行参数 | scripts/batch_extract.py |

修改后，应在 tests 中补充对应测试，并运行 python -m pytest，确认原有功能没有被破坏。

---

## 三、认识输入图片的命名规则

默认情况下，多张重复图片会自动合并成一个样本。

推荐文件名：

~~~text
BJC-001_rep1.jpg
BJC-001_rep2.jpg
BJC-001_rep3.jpg
~~~

以上图片会被识别为同一个 sample_id：BJC-001。

下面这种命名也可以：

~~~text
BJC-001_1.jpg
BJC-001_2.jpg
~~~

但如果名字没有明显的重复编号，例如 sample_A1.jpg，程序会保留完整文件名 sample_A1，避免错误地把不同样本合并。

### 自定义样本编号提取规则

如果文件名是：

~~~text
2026-BJC001-leaf-01.jpg
2026-BJC001-leaf-02.jpg
~~~

可以使用正则表达式，并把第一个括号内的内容作为 sample_id：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/result.csv --method exg --id-pattern '(\d{4}-BJC\d+)'
~~~

如果不想合并重复图片，添加：

~~~bash
--no-aggregate
~~~

完整示例：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/per_image.csv --method exg --no-aggregate
~~~

---

## 四、推荐的配置文件运行方式

命令行适合快速试用；正式项目建议保存一份 config.yaml，以便日后复现实验。

项目已经提供默认配置：

~~~bash
python scripts/batch_extract.py --config config.yaml --verbose
~~~

配置文件中的相对路径以 config.yaml 所在目录为准，不受当前终端目录影响。

第一次使用时，主要检查这几项：

~~~yaml
input:
  image_dir: "./data/raw_images/"
  output_dir: "./output/"

imaging:
  white_balance: "gray_world"

color_calibration:
  enabled: false

segmentation:
  method: "exg"
  device: "cpu"
  min_leaf_area_ratio: 0.002
  exclude_border_components: true
  border_margin_ratio: 0.01

output:
  format: "csv"
  phenotype_table_name: "leaf_color_phenotypes"
  separate_visualization: true
~~~

配置文件和命令行参数同时出现时，明确写在命令行里的参数优先。例如：

~~~bash
python scripts/batch_extract.py --config config.yaml --method grabcut --output output/grabcut_result.xlsx
~~~

这条命令只临时覆盖分割方法和输出文件，不会修改 config.yaml。

---

## 五、该选哪一种分割方法

| 方法 | 是否需要模型 | 是否需要显卡 | 适合场景 |
|---|---:|---:|---|
| exg | 否 | 否 | 第一次使用；绿色叶片与背景差异明显 |
| grabcut | 否 | 否 | 背景较复杂，但主要叶片位于画面中央 |
| auto | 可选 | 否 | 有 U-Net 模型就优先用模型，否则自动选择传统方法 |
| unet | 是 | 可选 | 已有同类图像训练出的 U-Net 权重 |
| sam | 是 | 通常建议 | 已安装 segment-anything 并准备好 SAM 检查点 |

### 推荐尝试顺序

1. 先用 exg，并加上 --visualize 检查结果。
2. 如果叶片漏分较多，尝试 grabcut。
3. 如果传统方法在不同背景下不稳定，再标注数据并训练 U-Net。

ExG：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/exg.csv --method exg --visualize
~~~

GrabCut：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/grabcut.csv --method grabcut --visualize
~~~

自动模式：

~~~bash
python scripts/batch_extract.py --config config.yaml --method auto
~~~

U-Net：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/unet.csv --method unet --model models/unet_cabbage.pth --device cpu --visualize
~~~

有可用的 NVIDIA CUDA 环境时，可把 --device cpu 改成 --device cuda。

### 如果确实要使用 SAM

SAM 不会随 requirements.txt 自动安装，需要单独安装：

~~~bash
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
~~~

再准备与模型类型匹配的检查点。下面以 ViT-H 检查点为例：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/sam.csv --method sam --model models/sam_vit_h.pth --device cpu --visualize
~~~

使用其他 SAM 模型类型时，在 config.yaml 中同时写明 model_type 和检查点：

~~~yaml
segmentation:
  method: "sam"
  model_type: "vit_b"
  model_path: "./models/sam_vit_b.pth"
  device: "cpu"
~~~

---

## 六、如何训练 U-Net

只有在传统分割效果不够稳定时，才需要这一步。

### 1. 准备训练图片和掩膜

~~~text
data/train/
├── images/
│   ├── leaf001.jpg
│   ├── leaf002.jpg
│   └── ...
└── masks/
    ├── leaf001.png
    ├── leaf002.png
    └── ...
~~~

要求：

- 图片和掩膜的基本文件名必须一致。
- 掩膜中叶片区域为白色或非零，背景为黑色或零。
- 当前训练脚本读取文件夹配对数据，不直接读取 COCO 或 LabelMe 标注文件。

### 2. 安装完整依赖

~~~bash
python -m pip install -r requirements.txt
~~~

### 3. 开始训练

有 CUDA：

~~~bash
python scripts/train_segmentation.py --images data/train/images --masks data/train/masks --backbone efficientnet-b3 --epochs 100 --device cuda --output models/unet_cabbage.pth
~~~

只使用 CPU：

~~~bash
python scripts/train_segmentation.py --images data/train/images --masks data/train/masks --backbone efficientnet-b3 --epochs 100 --device cpu --output models/unet_cabbage.pth
~~~

CPU 训练通常很慢，建议先用较少图片和较少轮数确认流程可运行。

### 4. 用训练好的模型批量提取

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/unet_result.csv --method unet --model models/unet_cabbage.pth --device cpu --visualize
~~~

---

## 七、白平衡怎么选

### gray_world：默认推荐

~~~bash
--white-balance gray_world
~~~

不需要额外数据，适合多数普通拍摄场景。

### none：完全不校正

~~~bash
--white-balance none
~~~

只建议在图片已经完成统一、可靠的白平衡校正时使用。

### perfect_reflector：高光反射法

~~~bash
--white-balance perfect_reflector
~~~

适合画面中存在接近白色区域的图片，但容易受过曝点影响。

### gray_card：灰卡法

灰卡法必须在 config.yaml 中提供灰卡区域的平均 RGB 值：

~~~yaml
imaging:
  white_balance: "gray_card"
  gray_card_rgb: [0.42, 0.40, 0.38]
~~~

三个值按 R、G、B 顺序填写，建议归一化到 0 到 1。

如果只在命令行写 --white-balance gray_card，却没有提供 gray_card_rgb，程序会明确报错。

---

## 八、颜色校准是否需要开启

### 不需要开启的情况

- 只是想先跑通流程。
- 图片来自同一相机、同一光源和同一批次。
- 当前主要目标是比较同批样本的相对差异。

此时保持：

~~~yaml
color_calibration:
  enabled: false
~~~

### 建议开启的情况

- 需要比较不同日期、不同相机或不同地点拍摄的数据。
- 对绝对颜色值有较高要求。
- 每批图片都拍摄了标准 ColorChecker 色卡。

### 颜色校准的实际操作

当前项目不会自动定位色卡色块。你需要使用图像软件或其他色卡识别工具，按参考色卡的固定顺序提取 24 个色块的平均 RGB。

保存为 measured_rgb.npy，形状必须是 (24, 3)，数值应为 0 到 1。然后运行：

~~~python
from pathlib import Path

import numpy as np

from src.preprocessing import ImagePreprocessor

measured = np.load("measured_rgb.npy").astype(np.float32)
if measured.max() > 1.0:
    measured = measured / 255.0

if measured.shape != (24, 3):
    raise ValueError("measured_rgb.npy 必须是 24 行、3 列")

calibrator = ImagePreprocessor(
    calibration_method="polynomial",
    polynomial_degree=2,
)
matrix = calibrator.compute_color_correction_matrix(
    measured_rgb=measured,
)

Path("models").mkdir(parents=True, exist_ok=True)
np.save("models/colorchecker_ccm.npy", matrix)
~~~

然后修改 config.yaml：

~~~yaml
color_calibration:
  enabled: true
  method: "polynomial"
  degree: 2
  ccm_file: "./models/colorchecker_ccm.npy"
~~~

如果 enabled 为 true，但没有提供有效的 matrix 或 ccm_file，程序会在启动时停止并说明原因，不会悄悄跳过校准。

支持的 CCM 文件格式包括 NPY、JSON、YAML 和 CSV。

---

## 九、输出文件怎么看

### 1. 每张图片的基础特征

常见列包括：

- sample_id：样本编号。
- image_path：原始图片路径。
- CIELAB_L_mean、CIELAB_a_mean、CIELAB_b_mean：叶片区域的 CIELAB 均值。
- HSV_H_mean、HSV_S_mean、HSV_V_mean：HSV 颜色统计。
- ExG、ExR、GLI、NGRDI、VARI、CIVE、DGCI、COM：植被指数。
- GLCM、LBP、Gabor：纹理特征。
- QC_mask_area_px：叶片掩膜像素数。
- QC_mask_area_ratio：叶片面积占整张图片的比例。
- QC_empty_mask：是否没有分割出叶片。
- QC_bbox_x、QC_bbox_y、QC_bbox_w、QC_bbox_h：叶片外接框。

实际列数会随 config.yaml 中启用的特征组变化。

### 2. 重复图片聚合后的列名

假设原始单图列为 CIELAB_L_mean：

- CIELAB_L_mean：重复图片之间的均值，保留原列名。
- CIELAB_L_mean_rep_std：重复图片之间的标准差。
- CIELAB_L_mean_rep_cv：重复图片之间的变异系数。
- n_replicates：该样本包含的图片数量。

这里要区分：

- 原列名中的 mean 或 std，是单张图片内部叶片像素的统计量。
- rep_std 和 rep_cv，是同一样本多张重复图片之间的统计量。

当重复均值非常接近 0 时，rep_cv 会留空，避免产生无意义的极大值或无穷值。

### 3. 输出为 Excel 或 JSON

Excel：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/result.xlsx --method exg
~~~

JSON：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/result.json --method exg
~~~

CSV：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/result.csv --method exg
~~~

程序根据输出文件扩展名选择格式。

---

## 十、如何判断一批数据是否处理成功

建议按下面顺序检查：

1. 终端最后是否显示成功和失败图片数量。
2. 输出表中行数是否与预期样本数一致。
3. n_replicates 是否与每个样本的重复图片数一致。
4. 随机抽查 visualizations 中至少 10% 的图片。
5. 查看 QC_empty_mask 是否为 True。
6. 检查 QC_mask_area_ratio 是否异常小或异常大。
7. 查看 rep_cv 特别大的样本，并回到原图检查。

如果部分图片失败，程序会生成：

~~~text
output/leaf_color_phenotypes_failures.csv
~~~

其中会记录失败图片和错误原因。

默认情况下，只要有图片失败，命令就会返回非零状态，方便自动化流程发现问题。确认可以接受不完整结果时，才添加：

~~~bash
--allow-partial
~~~

例如：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/result.csv --method exg --allow-partial
~~~

---

## 十一、常见问题

### 问题 1：提示 No images found

检查：

- 当前是否位于 leaf_color_phenotyping 目录。
- --input 后面的目录是否真实存在。
- 图片扩展名是否受支持。
- config.yaml 中的相对路径是否写对。

可以先执行：

~~~powershell
Get-ChildItem data\raw_images
~~~

macOS 或 Linux：

~~~bash
ls data/raw_images
~~~

### 问题 2：提示 ModuleNotFoundError

确认虚拟环境已启用，然后重新安装基础依赖：

~~~bash
python -m pip install numpy opencv-python scikit-image pandas PyYAML openpyxl
~~~

如果报错来自 torch、segmentation_models_pytorch 或 albumentations，说明正在使用 U-Net 相关功能，需要：

~~~bash
python -m pip install -r requirements.txt
~~~

### 问题 3：U-Net 提示找不到模型

检查 --model 路径是否存在。尚未训练模型时，先改用：

~~~bash
--method exg
~~~

### 问题 4：CUDA 不可用

把设备改成 CPU：

~~~bash
--device cpu
~~~

### 问题 5：叶片和背景分不开

依次尝试：

1. 加 --visualize 确认问题位置。
2. 从 exg 改为 grabcut。
3. 改善拍摄背景和光照。
4. 标注一批同类图片并训练 U-Net。

### 问题 6：开启颜色校准后立刻报错

这是保护机制。请确认：

- ccm_file 指向真实文件。
- 文件内矩阵形状正确。
- 多项式校准的矩阵行数与 degree 匹配。
- measured RGB 与参考色卡的 24 色顺序一致。

### 问题 7：Excel 打开 CSV 时路径或中文显示异常

直接输出 XLSX：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/result.xlsx --method exg
~~~

### 问题 8：PowerShell 不允许启用虚拟环境

只对当前 PowerShell 会话临时放行：

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
~~~

---

## 十二、正式用于 GWAS 前的建议

建议保留两个版本的数据：

1. 使用 --no-aggregate 导出的逐图片数据，用于检查重复间差异。
2. 默认聚合后的逐样本数据，用于与基因型和试验设计表合并。

R 中可先读取结果：

~~~r
library(data.table)

pheno <- fread("output/leaf_color_phenotypes.csv")
geno  <- fread("genotypes.csv")
meta  <- fread("experimental_metadata.csv")

dat <- merge(pheno, meta, by = "sample_id")
dat <- merge(dat, geno, by = "sample_id")
~~~

批次、区组、时期、地点和处理等实验信息应来自你的试验设计表，再按 sample_id 合并。不要把文件名解析当成完整的试验设计数据。

正式分析前还应：

- 保存原始图片，不覆盖。
- 保存本次 config.yaml。
- 记录相机、镜头、光源和曝光设置。
- 不同拍摄批次分别检查白平衡与颜色校准。
- 根据 QC 列和分割图排除失败图片。
- 检查极端值和重复间变异。

---

## 十三、运行自动测试

如果你修改了代码，建议先安装测试依赖：

~~~bash
python -m pip install -r requirements-dev.txt
~~~

然后运行：

~~~bash
python -m pytest
~~~

测试覆盖颜色空间转换、植被指数、分割基础用例、样本聚合、配置初始化和输出格式等关键路径。

---

## 十四、项目目录说明

~~~text
leaf_color_phenotyping/
├── config.yaml                 # 主配置文件
├── requirements.txt            # 完整运行和训练依赖
├── requirements-dev.txt        # 轻量测试依赖
├── scripts/
│   ├── batch_extract.py        # 批量提取入口
│   └── train_segmentation.py   # U-Net 训练入口
├── src/
│   ├── pipeline.py             # 主处理流程
│   ├── segmentation.py         # ExG、GrabCut、U-Net 和 SAM
│   ├── preprocessing.py        # 图片读取、白平衡和颜色校准
│   ├── color_features.py       # 颜色特征
│   ├── vegetation_indices.py   # 植被指数
│   ├── texture_features.py     # 纹理特征
│   └── utils.py                # 文件、颜色空间和统计工具
├── tests/                      # 自动测试
├── data/
│   ├── raw_images/             # 待处理图片
│   └── train/                  # 可选训练数据
├── models/                     # 可选模型和 CCM
└── output/                     # 输出表和可视化
~~~

---

## 十五、最常用命令速查

最简单的 CPU 处理：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/result.csv --method exg
~~~

同时保存分割图：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/result.csv --method exg --visualize
~~~

使用配置文件：

~~~bash
python scripts/batch_extract.py --config config.yaml --verbose
~~~

保留每张图片，不聚合：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/per_image.csv --method exg --no-aggregate
~~~

使用 U-Net：

~~~bash
python scripts/batch_extract.py --input data/raw_images --output output/unet.csv --method unet --model models/unet_cabbage.pth --device cpu
~~~

运行测试：

~~~bash
python -m pytest
~~~

---

## 许可证

本项目使用 MIT License，详见 LICENSE。
