# 大白菜叶色表型提取工具

这个项目从叶片图像中自动分割叶片，并提取颜色、植被指数、纹理、形状和分割质量控制指标。它支持普通图片和相机 RAW 文件，可用于批量表型统计、品种比较和后续 GWAS 数据整理。

如果你是第一次使用，先完成“快速开始”。确认绿色轮廓只围住叶片后，再按“完整运行步骤”处理正式数据。

## 目录

1. [先了解三个重要规则](#一先了解三个重要规则)
2. [快速开始](#二快速开始)
3. [完整运行步骤](#三完整运行步骤)
4. [命令行参数](#四命令行参数)
5. [config.yaml 配置说明](#五configyaml-配置说明)
6. [分割方法怎么选择](#六分割方法怎么选择)
7. [输出文件和字段](#七输出文件和字段)
8. [Python 函数与类的参数](#八python-函数与类的参数)
9. [训练 U-Net 分割模型](#九训练-unet-分割模型)
10. [颜色校准](#十颜色校准)
11. [常见问题](#十一常见问题)
12. [测试与项目结构](#十二测试与项目结构)

---

## 一、先了解三个重要规则

### 1. 正式分析时，一张图最好只有一片主要叶片

项目可以分割一张图中的多个叶片，但不同特征对多叶片的处理方式不同：

- 颜色、植被指数和纹理特征使用掩膜中的全部叶片像素。
- `Shape_*` 形状特征只使用面积最大的轮廓。
- 可视化左上角的 `Area` 也是最大轮廓面积，不是所有轮廓面积之和。

因此，正式表型分析建议先把图像裁成“一张图一片叶”。

### 2. 同一批正式数据不要混用 RAF 和 JPG

RAF 和相机直出的 JPG 即使来自同一次拍摄，也可能存在不同的白平衡、降噪、锐化、色调曲线和压缩处理。项目会把 RAF 转成标准 sRGB，但无法完全复制相机内部生成 JPG 的处理流程。

建议：

- 正式数据全部使用 RAF，或者全部使用 JPG。
- 拍摄参数、光源、背景、距离和相机角度保持一致。
- RAF 与 JPG 的比较只用于测试，不要混入同一份正式统计结果。

### 3. 可视化中“看得见”不等于“被识别”

`output/visualizations/*_vis.png` 是“原图 + 绿色分割轮廓”。标尺、标签和背景仍会显示在原图里，只有被绿色轮廓包围的区域才进入特征计算。

检查时重点看绿色线：

- 绿色线只围住叶片：正确。
- 标尺也出现绿色线：标尺被错误识别。
- 叶片没有绿色线或轮廓缺失：叶片漏分割。

---

## 二、快速开始

以下命令默认在项目目录 `leaf_color_phenotyping` 中执行。

### Windows PowerShell

#### 第 1 步：进入项目目录

```powershell
cd C:\Users\你的用户名\项目位置\wcybigbaicai\leaf_color_phenotyping
```

#### 第 2 步：创建并启用虚拟环境

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

如果命令行开头已经出现 `(.venv)`，说明环境已启用。

#### 第 3 步：安装依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

读取 `.RAF`、`.CR2`、`.NEF`、`.ARW` 等 RAW 文件需要 `rawpy`。它已经写在 `requirements.txt` 中。

#### 第 4 步：放入图片

把图片放进：

```text
data/raw_images/
```

支持的扩展名：

```text
.jpg .jpeg .png .tif .tiff .bmp .raw .dng .cr2 .nef .arw .raf
```

扩展名大小写都可以，例如 `.RAF` 和 `.raf` 都能识别。

#### 第 5 步：首次运行

```powershell
python scripts\batch_extract.py --no-aggregate --visualize --verbose
```

脚本会自动加载项目根目录的 `config.yaml`。启动信息中应出现：

```text
Loaded config from: ...\leaf_color_phenotyping\config.yaml
Segmentation: auto
```

#### 第 6 步：查看结果

```text
output/leaf_color_phenotypes.csv
output/visualizations/图片名_vis.png
```

第一次运行一定要打开可视化图片，确认绿色轮廓只围住叶片。

### macOS / Linux

```bash
cd /path/to/wcybigbaicai/leaf_color_phenotyping
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/batch_extract.py --no-aggregate --visualize --verbose
```

---

## 三、完整运行步骤

### 步骤 1：统一拍摄条件

正式实验中，尽量固定：

- 相机、镜头、焦距和拍摄距离；
- 光源类型、亮度和照射方向；
- 背景颜色；
- 曝光、ISO、光圈和快门；
- 白平衡设置；
- 叶片朝向和发育时期。

建议把色卡或灰卡拍在单独的参考图中。不要让标尺贴住叶片，否则两者可能连成一个轮廓，边缘过滤将无法只删除标尺。

### 步骤 2：整理图片和文件名

推荐目录：

```text
data/raw_images/
├── BJC-001_rep1.RAF
├── BJC-001_rep2.RAF
├── BJC-002_rep1.RAF
└── BJC-002_rep2.RAF
```

默认样本编号解析规则：

| 文件名 | 得到的 `sample_id` |
|---|---|
| `BJC-001_rep1.RAF` | `BJC-001` |
| `BJC-001_repeat2.jpg` | `BJC-001` |
| `BJC-001_r3.png` | `BJC-001` |
| `BJC-001_2.jpg` | `BJC-001` |
| `sample_A1.jpg` | `sample_A1` |

注意：`L1.RAF` 和 `L1.jpg` 的 `sample_id` 都是 `L1`。默认聚合时，它们会被当作同一样本的重复并求平均。比较两种格式时应使用 `--no-aggregate`，或者放在不同目录中。

### 步骤 3：检查 `config.yaml`

当前推荐的关键设置是：

```yaml
imaging:
  white_balance: "gray_world"

segmentation:
  method: "auto"
  device: "cpu"
  grabcut_iterations: 5
  morph_kernel_size: 5
  min_leaf_area_ratio: 0.002
  exclude_border_components: true
  border_margin_ratio: 0.01

output:
  format: "csv"
  separate_visualization: false
  phenotype_table_name: "leaf_color_phenotypes"
```

这组参数适用于当前测试图：标尺靠近画面边缘，叶片位于画面内部，而且叶片面积相对整图较小。

### 步骤 4：先做小批量检查

先只放 2～5 张有代表性的图片，然后运行：

```powershell
python scripts\batch_extract.py --no-aggregate --visualize --verbose
```

这里使用 `--no-aggregate`，是为了让每张图片在 CSV 中保留一行，便于逐图排查。

### 步骤 5：检查分割质量

打开 `output/visualizations/` 中的图片，逐张检查：

- 叶片边缘是否完整；
- 叶柄是否保留；
- 标尺、标签、背景是否没有绿色轮廓；
- 小叶片是否漏掉；
- 阴影是否被误认为叶片。

同时检查 CSV 中的质量控制列：

- `QC_mask_area_px`：所有掩膜像素总数；
- `QC_mask_area_ratio`：掩膜占整张图的比例；
- `QC_mask_is_empty`：`1` 表示没有识别到叶片；
- `QC_bbox_x/y/width/height`：所有掩膜的总包围框。

如果一批图片的拍摄距离一致，`QC_mask_area_ratio` 突然明显偏大通常表示背景或标尺被识别；突然接近 0 通常表示漏分割。

### 步骤 6：调整分割参数

常用调整方向：

| 问题 | 建议调整 |
|---|---|
| 标尺贴近图像边缘并被识别 | 保持 `exclude_border_components: true`；增大 `border_margin_ratio`，如 `0.02` |
| 标尺没有接触边缘 | 仅靠边缘过滤无法删除；重新构图、裁图，或使用训练好的 U-Net |
| 小叶片被删除 | 减小 `min_leaf_area_ratio`，如从 `0.002` 改为 `0.001` |
| 小噪点过多 | 增大 `min_leaf_area_ratio` 或 `morph_kernel_size` |
| 叶片边缘损失 | 减小 `morph_kernel_size`；优先尝试 `auto` |
| 叶片和标尺连在一起 | 先裁图或改变摆放方式；连通域过滤无法拆开已连接的物体 |

每次调整后重新运行并检查绿色轮廓，不要只看 CSV 数字。

### 步骤 7：正式批量运行

确认分割稳定后，可以保留每张图片：

```powershell
python scripts\batch_extract.py --no-aggregate --visualize --verbose
```

也可以按 `sample_id` 汇总重复：

```powershell
python scripts\batch_extract.py --visualize --verbose
```

默认汇总会为每个数值性状生成：

- 原性状名：重复的均值；
- `*_rep_std`：重复间标准差；
- `*_rep_cv`：重复间变异系数；
- `n_replicates`：重复数量。

建议先保存不聚合的逐图结果完成质控，再生成聚合后的正式结果。

### 步骤 8：保存实验记录

正式分析至少保留：

- 原始图片；
- 本次使用的 `config.yaml`；
- 未聚合的逐图表型表；
- 聚合后的样本表型表；
- 分割可视化；
- 失败报告；
- 文件名与材料编号、处理、重复、发育时期之间的对应表。

---

## 四、命令行参数

批量入口：

```powershell
python scripts\batch_extract.py [参数]
```

| 参数 | 默认行为 | 说明 |
|---|---|---|
| `--config`, `-c` | 项目根目录 `config.yaml` | 指定 YAML 配置文件。相对路径以当前终端目录解析，配置内部路径以配置文件目录解析 |
| `--input`, `-i` | 读取配置中的 `input.image_dir` | 临时覆盖输入图片目录，会递归查找子目录 |
| `--output`, `-o` | 根据配置生成输出路径 | 指定 `.csv`、`.json`、`.xlsx` 或 `.xls` 文件 |
| `--method`, `-m` | 读取配置 | 可选 `exg`、`grabcut`、`unet`、`sam`、`auto` |
| `--model` | 读取配置 | U-Net 或 SAM 权重文件路径 |
| `--device` | 读取配置，通常为 `cpu` | `cpu` 或 `cuda` |
| `--white-balance`, `-wb` | 读取配置 | `gray_world`、`perfect_reflector`、`gray_card` 或 `none` |
| `--id-pattern` | 自动解析 | 自定义样本编号正则表达式，第一个捕获组必须是样本 ID |
| `--no-aggregate` | 默认聚合 | 每张图片保留一行，不按 `sample_id` 求均值 |
| `--visualize` | 读取 `separate_visualization` | 保存“原图 + 绿色轮廓 + 关键指标”图片 |
| `--verbose`, `-v` | 关闭批次进度 | 显示发现的图片数和逐图进度 |
| `--allow-partial` | 有失败时退出码为 2 | 即使部分图片失败，也以成功退出码结束；失败仍写入报告 |

常用示例：

```powershell
# 推荐的首次检查
python scripts\batch_extract.py --no-aggregate --visualize --verbose

# 指定另一批图片和输出表
python scripts\batch_extract.py --input data\experiment_01 --output output\experiment_01.xlsx --no-aggregate --visualize

# 临时切换为 ExG
python scripts\batch_extract.py --method exg --output output\exg_check.csv --no-aggregate --visualize

# 使用自定义样本编号规则
python scripts\batch_extract.py --id-pattern "(BJC-\d+)" --no-aggregate

# 显式指定另一份配置
python scripts\batch_extract.py --config configs\experiment_02.yaml --verbose
```

命令行参数只覆盖本次运行，不会修改 `config.yaml`。

---

## 五、`config.yaml` 配置说明

### `input`：输入与输出路径

| 字段 | 当前默认值 | 说明 |
|---|---|---|
| `image_dir` | `./data/raw_images/` | 输入图片根目录；批处理会递归搜索 |
| `colorchecker_dir` | `./data/colorchecker/` | 色卡参考图目录；当前批处理不会自动从这里计算 CCM |
| `output_dir` | `./output/` | 未通过 `--output` 指定文件时使用 |
| `model_dir` | `./models/` | 模型目录记录项；实际模型路径由 `segmentation.unet_model` 或命令行指定 |

配置中的相对路径以配置文件所在目录为基准。

### `imaging`：成像预处理

| 字段 | 当前默认值 | 说明 |
|---|---|---|
| `color_space` | `sRGB` | 目标颜色空间说明项；读取函数当前固定输出 sRGB |
| `white_balance` | `gray_world` | 白平衡方法 |
| `gray_card_rgb` | 未设置 | `gray_card` 模式必填，格式为 `[R, G, B]`，范围 `[0,1]` |
| `bits_per_channel` | `16` | 配置说明项；RAW 读取接口当前默认使用 16 bit 后处理 |

白平衡方法：

| 方法 | 适用情况 | 注意事项 |
|---|---|---|
| `gray_world` | 一般批量拍摄，默认推荐 | 假设整幅图平均颜色接近中性灰；大面积单色背景可能影响结果 |
| `perfect_reflector` | 图中存在可靠高亮白区域 | 使用各通道高百分位值归一化 |
| `gray_card` | 有人工测得的灰卡 RGB | 最可控，但必须提供 `gray_card_rgb` |
| `none` | 图像已经完成统一、可靠的白平衡 | 不再做白平衡修正 |

### `color_calibration`：颜色校准矩阵

| 字段 | 默认值 | 说明 |
|---|---|---|
| `enabled` | `false` | 是否启用 CCM；没有经过验证的矩阵时不要开启 |
| `method` | `polynomial` | `linear` 或 `polynomial` |
| `polynomial_degree` | `2` | 多项式阶数；二阶矩阵应为 `9 × 3` |
| `ccm_file` | 空 | `.npy`、`.json`、`.yaml`、`.yml`、`.csv` 或空白分隔文本 |
| `matrix` | 未设置 | 也可直接在 YAML 中写矩阵，优先级高于 `ccm_file` |

`enabled: true` 但既没有 `matrix` 也没有 `ccm_file` 时，程序会立即报错，避免把未校准数据误当成已校准数据。

### `segmentation`：叶片分割

| 字段 | 默认值 | 说明 |
|---|---|---|
| `method` | `auto` | 分割方法 |
| `unet_model` | `models/unet_efficientnet_b3.pth` | U-Net 权重路径；内部映射为 `model_path` |
| `backbone` | 未写，代码默认 `efficientnet-b3` | U-Net 编码器，必须与训练时一致 |
| `device` | `cpu` | `cpu` 或 `cuda` |
| `exg_threshold` | `0.15` | 固定 ExG 阈值；只有 ExG 且 `use_otsu: false` 时才直接生效 |
| `use_otsu` | 未写，代码默认 `true` | ExG 是否使用 Otsu 自动阈值；`auto` 内部固定使用默认 Otsu 行为 |
| `grabcut_iterations` | `5` | GrabCut 迭代次数，内部映射为 `iterations` |
| `morph_kernel_size` | `5` | 形态学开闭运算核尺寸；应使用正整数，通常用奇数 |
| `min_leaf_area_ratio` | `0.002` | 小于整图面积 0.2% 的连通域被删除 |
| `exclude_border_components` | `true` | 删除进入边缘排除带的连通域 |
| `border_margin_ratio` | `0.01` | 边缘带宽占图像宽、高的比例 |
| `model_type` | 未写，SAM 默认 `vit_h` | SAM 模型类型，必须与检查点一致 |

`auto` 有一个重要行为：如果 `unet_model` 指向的文件真实存在，会优先创建 U-Net 分割器；文件不存在时使用 ExG 粗分割加 GrabCut 精修。

### `features`：特征开关

| 字段 | 说明 |
|---|---|
| `color_spaces` | 可选 `RGB`、`HSV`、`CIELAB`、`YCbCr` |
| `color_moments` | 是否计算 RGB 各通道均值、标准差、偏度和峰度 |
| `histogram.enabled` | 是否计算灰度及 RGB 直方图 |
| `histogram.bins` | 直方图分箱数；越大列数越多 |
| `histogram.percentiles` | 通道统计所需百分位，例如 `[10,25,50,75,90]` |
| `vegetation_indices.enabled` | 是否计算 RGB 植被指数 |
| `vegetation_indices.indices` | 指定指数名称列表 |
| `chromaticity.enabled` | 是否计算 CIE xyY 和 u'v' 色度坐标 |
| `chromaticity.spaces` | 当前实现中的说明项；实际开关由 `enabled` 控制 |
| `texture.enabled` | 是否计算 GLCM 纹理特征 |
| `texture.distances` | GLCM 像素距离 |
| `texture.angles` | 角度，单位为度 |
| `texture.properties` | `contrast`、`dissimilarity`、`homogeneity`、`energy`、`correlation`、`ASM` |
| `shape.enabled` | 是否计算形状特征 |
| `shape.features` | 需要输出的形状字段列表 |

### `output`：结果输出

| 字段 | 默认值 | 说明 |
|---|---|---|
| `format` | `csv` | `csv`、`json` 或 `excel` |
| `separate_visualization` | `false` | 是否默认保存可视化；命令行 `--visualize` 可临时开启 |
| `phenotype_table_name` | `leaf_color_phenotypes` | 未指定输出文件名时的表名 |

如果 `--output` 已带扩展名，扩展名优先决定格式。例如 `--output result.xlsx` 会输出 Excel，即使配置中写的是 CSV。

---

## 六、分割方法怎么选择

| 方法 | 是否需要模型 | 适用情况 | 局限 |
|---|---:|---|---|
| `auto` | 通常不需要 | 默认推荐；绿色叶片、暗背景、批量自动处理 | 颜色与叶片相近的物体仍可能误分 |
| `exg` | 否 | 光照稳定、绿色叶片与背景差异明显 | 对绿色标尺、反光和复杂背景较敏感 |
| `grabcut` | 否 | 主要目标位于图像中央 | CLI 批处理不能逐张手工给矩形；目标偏离中央时可能失败 |
| `unet` | 是 | 构图复杂、标尺位置变化大、传统方法不稳定 | 需要标注数据和匹配的模型权重 |
| `sam` | 是 | 单个主体接近图像中心的探索性分割 | 模型大、速度慢；当前批处理默认使用图像中心点提示 |

推荐顺序：

1. 先用当前 `auto` 配置。
2. 调整 `min_leaf_area_ratio`、边缘过滤和构图。
3. 如果标尺或标签经常位于图像内部，训练 U-Net。

---

## 七、输出文件和字段

### 表型表

默认输出：

```text
output/leaf_color_phenotypes.csv
```

主要字段组：

| 前缀或字段 | 含义 |
|---|---|
| `sample_id` | 从文件名或正则表达式得到的样本编号 |
| `image_path` | 原始图片路径；聚合后保留组内第一条路径 |
| `RGB_*` | RGB 均值、归一化比例和通道比值 |
| `HSV_*` | 色相、饱和度、明度统计 |
| `CIELAB_*` | L*、a*、b*、色度、色相角、绿度和黄度 |
| `YCbCr_*` | 亮度和色度统计 |
| `ColorMoment_*` | RGB 颜色矩 |
| `Hist_*` | 灰度和 RGB 直方图 |
| `Chromaticity_*` | CIE xyY、u'v' 色度坐标 |
| `VARI`、`GLI`、`DGCI` 等 | RGB 植被指数及其标准差、中位数 |
| `GLCM_*` | 纹理统计 |
| `Shape_*` | 最大叶片轮廓的形状特征 |
| `Uniformity_*` | CIELAB 颜色均匀性 |
| `QC_*` | 分割质量控制 |

常见 CIELAB 解释：

- `CIELAB_L_mean` 越大，叶片越亮。
- `CIELAB_A_mean` 越负，通常越偏绿。
- `CIELAB_greenness = -a*`，数值越大通常越绿。
- `CIELAB_B_mean` 越大，通常越偏黄。

这些指标受相机、光源、白平衡和曝光影响。跨批次比较时必须统一成像流程或使用可靠的颜色校准。

### 可视化

默认位置：

```text
output/visualizations/<原文件名>_vis.png
```

内容包括原图、绿色分割轮廓以及 L*、a*、b*、GLI、DGCI 和最大轮廓面积。当前实现不会生成单独的热力图或纯掩膜文件；需要纯掩膜时请通过 Python API 的 `process_single(..., return_visualization=True)` 获取。

### 失败报告

只要有图片处理失败，就会写入：

```text
output/leaf_color_phenotypes_failures.csv
```

字段包括：

- `image_path`；
- `error_type`；
- `error`。

没有 `--allow-partial` 时，成功图片仍可能写入结果表，但命令会以退出码 2 结束，提醒自动化流程不要把不完整结果当作完全成功。

---

## 八、Python 函数与类的参数

日常批量使用只需要命令行。需要在自己的脚本或 Notebook 中组合流程时，再使用下面的 Python API。

以下划线开头的方法，例如 `_init_segmenter()`、`_aggregate_by_sample()`、`_vari()`，属于内部实现，不建议由外部代码直接调用。

### 8.0 命令入口：`scripts.batch_extract`

| 函数 | 参数和返回值 | 说明 |
|---|---|---|
| `parse_args()` | 从 `sys.argv` 读取参数，返回 `argparse.Namespace` | 定义第四章列出的全部批处理命令行参数；默认配置路径为项目根目录 `config.yaml` |
| `main()` | 无显式参数，返回整数退出码 | `0` 表示成功，`1` 表示没有得到任何表型，`2` 表示存在失败图片且未使用 `--allow-partial` |

### 8.1 完整流水线：`src.pipeline`

#### `LeafColorPipeline(config=None)`

| 参数 | 类型/默认值 | 说明 |
|---|---|---|
| `config` | `dict` 或 `None` | 完整配置字典。命令行会从 YAML 读取；直接传 `None` 时使用各模块的代码默认值 |

```python
from pathlib import Path
import yaml
from src.pipeline import LeafColorPipeline

config_path = Path("config.yaml").resolve()
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["_config_dir"] = str(config_path.parent)
pipeline = LeafColorPipeline(config)
```

#### `process_single(...)`

```python
process_single(
    image_path,
    sample_id=None,
    replicate=None,
    developmental_stage=None,
    metadata=None,
    white_balance=None,
    gray_roi=None,
    return_visualization=False,
)
```

| 参数 | 说明 |
|---|---|
| `image_path: str` | 单张图片路径 |
| `sample_id: str or None` | 手工指定样本 ID；不填时从文件名解析 |
| `replicate: str or None` | 重复编号，仅作为元数据保存 |
| `developmental_stage: str or None` | 发育阶段，仅作为元数据保存 |
| `metadata: dict or None` | 其他元数据，例如 `{"treatment": "control"}` |
| `white_balance: str or None` | 临时覆盖白平衡；不填时使用配置 |
| `gray_roi: ndarray(3,) or None` | 灰卡平均 RGB，`gray_card` 模式必需 |
| `return_visualization: bool` | 为 `True` 时额外返回 `mask` 和 `visualization` |

返回字典包含 `sample_id`、`features`、`image_path`；启用可视化时还包含二值 `mask` 和 RGB 可视化数组。

```python
result = pipeline.process_single(
    "data/raw_images/L2.RAF",
    sample_id="L2",
    return_visualization=True,
)
print(result["features"]["CIELAB_L_mean"])
mask = result["mask"]
```

#### `process_batch(...)`

```python
process_batch(
    image_dir,
    output_dir=None,
    output_csv=None,
    id_pattern=None,
    group_by_sample=True,
    white_balance=None,
    gray_roi=None,
    save_visualizations=None,
    visualization_dir=None,
    verbose=True,
)
```

| 参数 | 说明 |
|---|---|
| `image_dir: str` | 输入目录，会递归搜索支持的图片 |
| `output_dir: str or None` | 输出目录；`output_csv` 为空时使用 |
| `output_csv: str or None` | 历史命名，实际支持 CSV、JSON、XLSX、XLS |
| `id_pattern: str or None` | 样本 ID 正则表达式，第一个捕获组为 ID |
| `group_by_sample: bool` | 是否按 `sample_id` 汇总 |
| `white_balance: str or None` | 批次白平衡覆盖值 |
| `gray_roi: ndarray(3,) or None` | 灰卡平均 RGB |
| `save_visualizations: bool or None` | `None` 表示读取配置；布尔值可直接覆盖 |
| `visualization_dir: str or None` | 自定义可视化目录 |
| `verbose: bool` | 是否显示批次发现和逐图进度 |

返回 `pandas.DataFrame`。失败记录保存在 `pipeline.last_batch_failures`。

### 8.2 图像预处理：`src.preprocessing`

#### `ImagePreprocessor(...)`

```python
ImagePreprocessor(
    target_illuminant="D65",
    calibration_method="polynomial",
    polynomial_degree=2,
)
```

| 参数 | 说明 |
|---|---|
| `target_illuminant` | 目标光源名称；当前主要作为对象设置保存 |
| `calibration_method` | `linear` 或 `polynomial` |
| `polynomial_degree` | 多项式 CCM 阶数 |

#### 公开方法

| 函数 | 关键参数 | 返回值与用途 |
|---|---|---|
| `read_raw(raw_path, use_camera_wb=True, output_bps=16, linear_output=False)` | RAW 路径；是否使用相机白平衡；8/16 bit；是否输出线性 sRGB | 返回 `float32 RGB [0,1]`。默认使用相机白平衡、标准 sRGB gamma、关闭自动增亮 |
| `read_image(image_path)` | 普通图片路径 | 返回 `float32 RGB [0,1]` |
| `white_balance_gray_world(img_rgb)` | RGB 数组 | 灰度世界白平衡后的 RGB |
| `white_balance_perfect_reflector(img_rgb, percentile=99.9)` | RGB 数组和高亮百分位 | 完美反射法白平衡结果 |
| `white_balance_gray_card(img_rgb, gray_roi)` | RGB 数组；形状为 `(3,)` 的灰卡均值 | 灰卡白平衡结果 |
| `set_color_correction_matrix(matrix)` | 线性 `3×3` 或对应阶数的多项式矩阵 | 校验并保存 CCM |
| `load_color_correction_matrix(path)` | `.npy/.json/.yaml/.csv/文本` | 读取、校验并保存 CCM |
| `compute_color_correction_matrix(measured_rgb, reference_lab=None)` | 测得的 `N×3` 色块 RGB；参考 Lab | 最小二乘计算并保存 CCM |
| `apply_color_correction(img_rgb)` | `float RGB [0,1]` | 应用当前 CCM；未设置 CCM 时返回原图并警告 |
| `process(image_path, white_balance_method="gray_world", gray_roi=None, apply_ccm=True)` | 图片、白平衡、灰卡值、是否应用 CCM | 返回 `rgb`、`rgb_uint8`、`hsv`、`lab` 字典 |
| `has_color_correction_matrix` | 只读属性 | 是否已经加载或计算 CCM |

### 8.3 叶片分割：`src.segmentation`

所有分割器共同使用这些后处理参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `morph_kernel_size` | `5` | 开闭运算核大小 |
| `min_area_ratio` | 传统方法 `0.002`，U-Net/SAM `0.005` | 删除过小连通域 |
| `exclude_border_components` | `False` | 是否删除边缘连通域；项目配置将其设为 `true` |
| `border_margin_ratio` | `0.01` | 边缘排除带比例 |

| 类或函数 | 特有参数 | 说明 |
|---|---|---|
| `BaseSegmenter(...)` | 只有共同参数 | 基类；`segment()` 需由子类实现，`postprocess(mask)` 做形态学和连通域过滤 |
| `ExGSegmenter(exg_threshold=0.15, use_otsu=True, ...)` | 固定阈值；是否使用 Otsu | `segment(img_rgb)` 返回二值掩膜；`compute_exg()` 和 `compute_exgr()` 返回指数图 |
| `GrabCutSegmenter(iterations=5, ...)` | GrabCut 迭代次数 | `segment(img_rgb, rect=None, init_mask=None)`；`rect=(x,y,w,h)`，`init_mask` 可提供初始前景 |
| `GrabCutSegmenter.segment_auto(img_rgb)` | 无额外参数 | 用 ExG 候选初始化 GrabCut；精修丢失叶片时退回 ExG 候选 |
| `UNetSegmenter(model_path, backbone="efficientnet-b3", device="cuda", ...)` | 权重、骨架、设备 | 延迟加载模型，预测阈值固定为 `0.5` |
| `SAMSegmenter(sam_checkpoint, model_type="vit_h", device="cuda", ...)` | 检查点、模型类型、设备 | `segment(img_rgb, center_point=None)`；不传点时使用图像中心 |
| `AutoSegmenter(iterations=5, ...)` | GrabCut 迭代次数 | ExG 候选 + GrabCut；不需要模型 |
| `create_segmenter(method="auto", **kwargs)` | 方法名及构造参数 | 工厂函数；`auto` 在有效 U-Net 权重存在时优先使用 U-Net，否则使用 `AutoSegmenter` |
| `normalize_imagenet_rgb(img_rgb)` | RGB 数组 | 将 `[0,1]` 或 `[0,255]` 图像按 ImageNet 均值和标准差归一化 |

所有 `segment()` 方法都返回 `(H,W)` 的 `uint8` 二值掩膜，背景为 `0`，叶片为 `255`。

### 8.4 颜色特征：`src.color_features`

#### `ColorFeatureExtractor(...)`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `color_spaces` | `RGB, HSV, CIELAB, YCbCr` | 要处理的颜色空间 |
| `hist_bins` | `32` | 直方图分箱数 |
| `hist_percentiles` | `10,25,50,75,90` | 通道统计百分位 |
| `include_color_moments` | `True` | 是否计算颜色矩 |
| `include_histogram` | `True` | 是否计算直方图 |
| `include_chromaticity` | `True` | 是否计算 xyY 和 u'v' |

公开方法：

- `extract(img_rgb, mask)`：输入 RGB 图像和同尺寸二值掩膜，返回扁平特征字典。
- `leaf_color_difference(lab1_mean, lab2_mean)`：输入两个 `(3,)` Lab 均值，返回 `delta_E76`、`delta_L`、`delta_a` 和 `delta_b`。

### 8.5 植被指数：`src.vegetation_indices`

#### `VegetationIndexExtractor(indices=None)`

`indices=None` 表示计算全部支持的指数；传空列表表示全部关闭。支持：

| 指数 | 核心含义 |
|---|---|
| `VARI` | 可见光抗大气植被指数 |
| `GLI` | 绿叶指数 |
| `ExG` | 超绿指数 |
| `ExR` | 超红指数 |
| `ExGR` | ExG − ExR |
| `NGRDI` / `NDI` | 归一化绿红差异 |
| `DGCI` | 暗绿色指数 |
| `CIVE` | 植被提取颜色指数 |
| `MGRVI` | 改进绿红植被指数 |
| `RGBVI` | RGB 植被指数 |
| `VEG` | 可见光植被指数 |
| `COM` | 多指数的组合量 |

- `compute(img_rgb, mask=None)`：输入 RGB 图像和可选掩膜；每个指数返回均值、`_std` 和 `_median`。
- `estimate_chlorophyll_from_rgb(img_rgb, mask=None, calibration="liang_2015")`：可选 `liang_2015`、`wang_2014`、`hunt_2013`。这是跨作物经验公式，不在主流水线中默认使用；正式实验应以本地 SPAD 实测值重新校准。

### 8.6 纹理、形状和均匀性：`src.texture_features`

#### `GLCMTextureExtractor(...)`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `distances` | `[1,3,5]` | 共生矩阵像素距离 |
| `angles` | `[0,45,90,135]` | 角度，单位为度 |
| `levels` | `64` | 灰度量化级数；较小更快 |
| `properties` | 六种标准属性 | 传空列表关闭纹理输出 |

`compute(img_rgb, mask=None)` 返回每个属性的总体均值、标准差和各距离均值。空掩膜返回 NaN。

#### `LeafShapeExtractor(features=None)`

`features` 可选：`area`、`perimeter`、`circularity`、`eccentricity`、`solidity`、`extent`、`aspect_ratio`、`roundness`、`major_axis_length`、`minor_axis_length`。

`compute(mask, pixel_scale=None)` 的参数：

- `mask`：二值掩膜；如果有多个轮廓，只计算最大轮廓。
- `pixel_scale`：可选的 `mm/pixel`；提供后面积变成 `mm²`，周长和轴长变成 `mm`。

#### `ColorTextureAnalyzer.color_uniformity(lab_img, mask)`

输入 CIELAB 图像和掩膜，返回 L/a/b 标准差、变异系数以及像素到叶片平均 Lab 的 ΔE76 均值和标准差。

### 8.7 文件、颜色转换和统计工具：`src.utils`

| 函数 | 参数 | 返回值/说明 |
|---|---|---|
| `find_images(directory, extensions=...)` | 根目录和扩展名元组 | 递归返回排序后的 `Path` 列表 |
| `parse_sample_id(filename, pattern=None)` | 文件名；可选正则 | 返回样本 ID；自定义正则必须含捕获组 |
| `safe_mkdir(path)` | 目录路径 | 创建目录并返回 `Path` |
| `read_image_rgb(path, as_float=True)` | 文件路径；是否归一化 | 返回 RGB；浮点模式范围 `[0,1]` |
| `read_image_gray(path)` | 文件路径 | 返回单通道灰度图 |
| `write_image_rgb(path, img_rgb)` | 输出路径和 RGB 数组 | 自动创建父目录并保存，支持中文和空格路径 |
| `list_subdirs(directory)` | 根目录 | 返回直接子目录列表 |
| `rgb_to_xyz(img_rgb, illuminant="D65")` | sRGB `[0,1]` | 返回 XYZ 图像 |
| `xyz_to_lab(img_xyz)` | XYZ 图像 | 返回 CIELAB 图像 |
| `rgb_to_lab(img_rgb)` | sRGB `[0,1]` | 返回 CIELAB 图像 |
| `rgb_to_hsv(img_rgb)` | RGB `[0,1]` | 返回 HSV 图像 |
| `rgb_to_ycbcr(img_rgb)` | RGB `[0,1]` | 返回 YCbCr 图像 |
| `rgb_to_chromaticity_xyy(img_rgb)` | RGB 图像 | 返回 `x, y, Y` 三个数组 |
| `rgb_to_chromaticity_uv(img_rgb)` | RGB 图像 | 返回 `u_prime, v_prime` |
| `channel_stats(channel, percentiles=(...))` | 单通道数组和百分位 | 返回均值、标准差、范围、中位数、偏度、峰度和百分位 |
| `histogram_features(channel, bins=32)` | `[0,255]` 通道和分箱数 | 返回归一化 `hist_bin_*` |
| `get_colorchecker_lab_d65()` | 无 | 当前返回内置 ColorChecker D50 数据的近似值；严谨校准应使用验证参考值 |
| `delta_e_76(lab1, lab2)` | 两组 Lab | 返回 CIE76 色差 |
| `delta_e_94(lab1, lab2, k_L=1, k_C=1, k_H=1)` | Lab 和权重 | 返回 CIE94 色差 |
| `delta_e_2000(lab1, lab2)` | 两组 Lab | 返回 CIEDE2000；大图逐像素计算较慢 |
| `find_colorchecker_roi(img_rgb, target_size=(24,24))` | RGB 图像和目标尺寸 | 当前为预留接口，会警告并返回 `None`，尚未自动检测色卡 |

---

## 九、训练 U-Net 分割模型

只有当传统方法无法稳定排除标尺、标签或复杂背景时，才需要训练模型。

### 1. 准备标注数据

```text
data/train/
├── images/
│   ├── leaf_001.jpg
│   ├── leaf_002.jpg
│   └── ...
└── masks/
    ├── leaf_001.png
    ├── leaf_002.png
    └── ...
```

要求：

- 图片支持 JPG、JPEG、PNG、TIF、TIFF；训练数据加载器不直接读取 RAF。
- 掩膜与图片主文件名必须一致。
- 掩膜是单通道二值图：背景 `0`，叶片 `255`。
- 掩膜优先使用 PNG，也接受同名 JPG。
- 至少需要 2 对图像和掩膜；实际训练应准备更多且覆盖真实拍摄差异。

### 2. 开始训练

有 NVIDIA CUDA 环境：

```powershell
python scripts\train_segmentation.py `
  --images data\train\images `
  --masks data\train\masks `
  --backbone efficientnet-b3 `
  --image-size 512 512 `
  --batch-size 8 `
  --epochs 100 `
  --device cuda `
  --output models\unet_cabbage.pth
```

只有 CPU：

```powershell
python scripts\train_segmentation.py --images data\train\images --masks data\train\masks --device cpu --batch-size 2 --workers 0 --output models\unet_cabbage.pth
```

训练参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--images` | 必填 | 训练图片目录 |
| `--masks` | 必填 | 二值掩膜目录 |
| `--backbone` | `efficientnet-b3` | `efficientnet-b0/b3`、`resnet34/50`、`mobilenet_v2` |
| `--image-size H W` | `512 512` | 训练输入尺寸 |
| `--batch-size` | `8` | 显存不足时减小 |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `1e-3` | 初始学习率 |
| `--weight-decay` | `1e-4` | AdamW 权重衰减 |
| `--workers` | `4` | Windows 出现 DataLoader 问题时改为 `0` |
| `--device` | `cuda` | `cuda` 或 `cpu`；请求 CUDA 但不可用时会报错 |
| `--output` | `models/unet_leaf.pth` | 最佳 IoU 权重保存路径 |

训练按固定随机种子划分 80% 训练集和 20% 验证集，保存验证 IoU 最好的模型，并在同名 JSON 文件中保存训练历史。

### 3. 在配置中启用模型

```yaml
segmentation:
  method: "unet"
  unet_model: "models/unet_cabbage.pth"
  backbone: "efficientnet-b3"
  device: "cpu"
  min_leaf_area_ratio: 0.002
  exclude_border_components: true
  border_margin_ratio: 0.01
```

`backbone` 必须与训练命令一致，否则权重无法加载。

### 4. 训练模块中的公开对象

| 对象 | 参数与用途 |
|---|---|
| `LeafSegmentationDataset(images_dir, masks_dir, image_size=(512,512), augment=False, pairs=None)` | 读取同名图像/掩膜；`augment=True` 启用 Albumentations；`pairs` 可传已配对路径列表。`len(dataset)` 返回配对数，`dataset[idx]` 返回 `(image_tensor, mask_tensor)` |
| `DiceLoss(smooth=1.0)` | 二值分割 Dice 损失，`smooth` 防止除零；`forward(pred, target)` 接收模型 logits 和二值目标张量，返回标量损失 |
| `BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)` | BCE 与 Dice 的加权和；`forward(pred, target)` 使用相同形状的 logits 和目标张量，返回标量损失 |
| `compute_iou(pred, target, threshold=0.5)` | 对 logits 做 sigmoid 和阈值化后计算 IoU |
| `train(args)` | 使用命令行解析得到的参数对象执行完整训练 |
| `parse_args()` | 返回训练命令行参数 |

---

## 十、颜色校准

颜色校准不是“打开开关就自动完成”。当前项目能够加载、计算和应用 CCM，但自动寻找 ColorChecker 的函数仍是预留接口。

可靠流程是：

1. 在相同光源和相机设置下拍摄 ColorChecker。
2. 人工或使用专门工具得到各色块的实测 RGB。
3. 使用可靠参考 Lab 值计算 CCM。
4. 在独立色卡图上验证 ΔE。
5. 保存矩阵并在 `config.yaml` 中启用。

线性矩阵示例：

```yaml
color_calibration:
  enabled: true
  method: "linear"
  ccm_file: "models/ccm_linear.npy"
```

二阶多项式输入展开为：

```text
R, G, B, R², G², B², RG, RB, GB
```

因此二阶多项式 CCM 的形状必须是 `9 × 3`。

在没有经过独立验证的 CCM 时，保持：

```yaml
color_calibration:
  enabled: false
```

---

## 十一、常见问题

### RAF 可以读取，但标尺也出现了绿色轮廓

先确认启动时出现 `Loaded config from:`。当前脚本会默认加载项目配置，但如果使用了另一份配置，需要确认其中包含：

```yaml
segmentation:
  min_leaf_area_ratio: 0.002
  exclude_border_components: true
  border_margin_ratio: 0.01
```

如果标尺没有进入边缘带，边缘过滤不会删除它。此时应裁图、改变构图，或使用 U-Net。

### 标尺仍然能在可视化中看到

这是正常的。可视化保留完整原图，只在掩膜边缘画绿色线。标尺可见但没有绿色轮廓，表示它没有参与计算。

### RAF 正常，但相同内容的 JPG 结果不同

相机 JPG 通常已经应用厂商色彩风格、白平衡、降噪、锐化、色调曲线和压缩；RAW 转换路径不同。即使内容相同，像素值也不会完全一致。正式分析请固定一种格式。

### 提示没有找到图片

检查：

1. 是否从项目目录运行；
2. 启动时打印的 `Input:` 是否正确；
3. 图片扩展名是否受支持；
4. `config.yaml` 的 `input.image_dir` 是否正确。

也可显式指定：

```powershell
python scripts\batch_extract.py --input "D:\实验数据\叶片图片" --no-aggregate --visualize
```

### 没有生成可视化

配置默认 `separate_visualization: false`。运行时添加：

```powershell
python scripts\batch_extract.py --visualize
```

### 小叶片消失

把：

```yaml
min_leaf_area_ratio: 0.002
```

逐步减小到 `0.001` 或 `0.0005`。参数过小会保留更多噪点。

### 掩膜为空

查看 `QC_mask_is_empty`。尝试：

- 检查图像是否过暗或严重偏色；
- 临时使用 `--white-balance none` 比较；
- 尝试 `--method exg` 或 `--method auto`；
- 检查 `min_leaf_area_ratio` 是否过大；
- 确认目标没有被边缘过滤删除。

### U-Net 权重加载失败

常见原因：

- `unet_model` 路径错误；
- `backbone` 与训练时不一致；
- 权重文件不是该项目保存的 `state_dict`；
- `torch` 或 `segmentation-models-pytorch` 未正确安装。

### CUDA 不可用

改为：

```yaml
device: "cpu"
```

或命令行使用 `--device cpu`。CPU 能运行，但深度学习分割和训练会更慢。

### 同一样本只剩一行

默认会按 `sample_id` 聚合。需要逐图结果时添加 `--no-aggregate`。

### 处理部分失败但仍生成了 CSV

这是设计行为：成功图片会写入主表，失败图片写入 `*_failures.csv`。没有 `--allow-partial` 时命令返回非零退出码，提示结果不完整。

---

## 十二、测试与项目结构

### 运行测试

开发测试依赖：

```powershell
python -m pip install -r requirements-dev.txt
```

运行：

```powershell
python -m pytest -q
```

当前测试覆盖 RAF 路由、RAW 后处理参数、颜色数学、边缘连通域过滤、重复汇总、输出格式和批处理失败报告等关键行为。

### 项目结构

```text
leaf_color_phenotyping/
├── config.yaml
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── data/
│   ├── raw_images/              # 输入图片
│   ├── colorchecker/            # 可选色卡参考图
│   └── train/
│       ├── images/              # U-Net 训练图片
│       └── masks/               # U-Net 二值掩膜
├── models/                      # U-Net、SAM 或 CCM 文件
├── output/
│   ├── leaf_color_phenotypes.csv
│   └── visualizations/
├── scripts/
│   ├── batch_extract.py         # 批量提取命令
│   └── train_segmentation.py    # U-Net 训练命令
├── src/
│   ├── pipeline.py              # 主流水线
│   ├── preprocessing.py         # RAW、白平衡、颜色校准
│   ├── segmentation.py          # 叶片分割
│   ├── color_features.py        # 颜色特征
│   ├── vegetation_indices.py    # 植被指数
│   ├── texture_features.py      # 纹理、形状、均匀性
│   └── utils.py                 # 文件、色彩空间和统计工具
├── notebooks/
│   └── demo_pipeline.ipynb      # 探索性示例
└── tests/                       # 自动测试
```

`notebooks/demo_pipeline.ipynb` 是探索性示例，不是批处理所必需。其统计分析单元还会使用 `scikit-learn` 和 `statsmodels`；如果要运行这些单元，需要另行安装：

```powershell
python -m pip install scikit-learn statsmodels
```

### 最推荐的日常命令

检查每张图并保存可视化：

```powershell
python scripts\batch_extract.py --no-aggregate --visualize --verbose
```

确认质量后按样本汇总：

```powershell
python scripts\batch_extract.py --visualize --verbose
```

查看所有命令参数：

```powershell
python scripts\batch_extract.py --help
python scripts\train_segmentation.py --help
```

---

## 许可

本项目采用 MIT License。
