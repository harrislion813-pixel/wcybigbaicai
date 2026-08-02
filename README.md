# 大白菜叶色表型提取

项目代码和完整使用手册位于 `leaf_color_phenotyping` 目录。

## 第一次使用（推荐桌面版）

进入 `leaf_color_phenotyping` 文件夹后：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gui.txt
.\.venv\Scripts\python.exe app.py
```

首次安装后，以后只需运行最后一条命令。进入界面后，在“叶片分析”中选择图片
和结果文件夹，先预检，再开始分析。

CCM 颜色校准已经改为图形化向导：导入两张独立拍摄的 ColorChecker 24
色卡图即可自动检测、取样、拟合和验证，不需要手工制作 RGB CSV。

详细说明见 [桌面版使用指南](leaf_color_phenotyping/GUI_GUIDE.md)。

## [完整中文操作手册](leaf_color_phenotyping/README.md)

手册包括：

- Windows、macOS 和 Linux 快速开始；
- 从整理图片、配置参数到正式批量运行的完整步骤；
- RAF/JPG、白平衡、标尺过滤和分割质量检查；
- 所有命令行参数和 `config.yaml` 字段；
- Python 公共函数、类、参数和返回值；
- U-Net 训练、颜色校准、输出字段和常见问题。

## 命令行模式

命令行入口继续保留给开发、自动化和高级参数控制：

```powershell
cd leaf_color_phenotyping
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\batch_extract.py --no-aggregate --visualize --verbose
```

图片放入 `leaf_color_phenotyping/data/raw_images/`，结果生成在 `leaf_color_phenotyping/output/`。
