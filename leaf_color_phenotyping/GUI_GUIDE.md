# 桌面版使用指南

桌面版把叶片批处理、分割预检、颜色校准和 Profile 管理集中在一个窗口中。
所有计算仍使用原有 Python 流水线，命令行入口继续保留。

## 1. 安装和启动

当前版本暂未制作独立安装包，因此电脑仍需预先安装 Python 3.10 或更高版本，
首次安装依赖时需要联网。

Windows 用户在 PowerShell 中进入项目目录后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gui.txt
.\.venv\Scripts\python.exe app.py
```

macOS 或 Linux 用户执行：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-gui.txt
./.venv/bin/python app.py
```

依赖只需安装一次。以后启动时，Windows 运行：

```powershell
.\.venv\Scripts\python.exe app.py
```

直接调用虚拟环境中的 Python，不需要激活环境，也不受 CMD 文件关联或
PowerShell 脚本执行策略影响；启动错误会直接显示在当前终端中。

## 2. 叶片分析

1. 在“叶片分析”页选择图片文件夹和结果文件夹。结果文件夹必须位于图片文件夹之外，避免把生成的预览图再次当作原始图片。
2. 选择分析模式：
   - **同批次相对比较**：默认模式，不使用 CCM，适合同一固定拍摄批次内比较。
   - **跨批次/跨设备比较**：必须选择状态为 `validated` 的颜色 Profile。
3. 点击“预检代表图片”。程序会从整批图片的不同位置选择最多 5 张代表图。
4. 检查绿色轮廓是否只包围叶片；红色区域表示被排除的白色叶柄或主脉。
5. 点击“开始批量分析”。运行时可以安全取消，已经完成的结果仍会保存。任务运行中关闭窗口时，程序会先请求取消并等待工作线程安全结束。
6. 完成后在结果表中快速查看数据，或点击“打开结果文件夹”。

高级设置默认折叠。只有分割效果不理想时才需要调整分割方法、计算设备、
样本编号正则或白色组织过滤。

## 3. 创建 CCM 颜色 Profile

### 拍摄要求

- 准备一张训练色卡图和另一张真正独立拍摄的验证色卡图；重命名、重新编码或只修改色卡外区域都不算独立验证。
- 两张图保持相机、镜头、光源、曝光、白平衡和文件类型一致。
- 不要让色卡过曝、反光或被阴影覆盖。
- 同一组校准图必须全部使用 RAW，或者全部使用 JPG/PNG/TIFF。

### 界面操作

1. 打开“颜色校准”页，选择训练图和独立验证图。
2. 填写 Profile 名称和相机/拍摄方案 ID。
3. 根据实际色卡选择 2014 年 11 月前或之后的版本。
4. 点击“自动识别色块并创建 Profile”。
5. 检查两张取样预览图中的 24 个绿色方框。

自动识别失败时，点击“手工设置训练图四角”或“手工设置验证图四角”，
依次点击色卡左上、右上、右下和左下四个角。色卡占满整张图片时可以使用
“色卡占满整张图”。

程序会先比较两次拍摄实际提取的 24 色块数据；相同或近乎相同的采样会被拒绝。
随后自动比较线性 3×3 和 root-polynomial 模型。只有独立验证通过全部
ΔE00 门槛时，Profile 才会标记为 `validated`；未通过的结果仍会保存为
`draft`，用于排查，但不能用于跨批次分析。

Profile 旁边会同时保存两份 `*_patches.csv`，用于审计自动提取的 24 色块数据。

## 4. Profile 管理

“Profile 管理”页会扫描 `models` 中的 `.ccm.json` 文件，并显示：

- 验证状态；
- 相机和图片类型；
- 工作颜色域和白平衡；
- 验证集 ΔE00 中位数；
- 完整性错误。

只有通过完整性检查的 `validated` Profile 会出现在跨批次分析的选择框中。
选择 Profile 后，桌面版会自动锁定相机 ID、白平衡、RAW 解码和曝光处理信息，
避免运行条件与校准条件不一致。

## 5. 输出文件

一次完整运行通常生成：

```text
leaf_color_phenotypes.csv/xlsx/json
leaf_color_phenotypes_raw.csv/xlsx/json
leaf_color_phenotypes_manifest.json
leaf_color_phenotypes_failures.csv       # 仅有失败图片时
visualizations/                          # 启用预览时
```

manifest 会记录最终配置、依赖版本、校准 Profile 哈希、成功/失败数量以及运行是
完成还是被取消。正式分析时应与表型表一起归档。
