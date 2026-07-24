# 大白菜叶色表型提取

项目代码和完整使用手册位于 `leaf_color_phenotyping` 目录。

## 第一次使用

请从这份文档开始：

## [完整中文操作手册](leaf_color_phenotyping/README.md)

手册包括：

- Windows、macOS 和 Linux 快速开始；
- 从整理图片、配置参数到正式批量运行的完整步骤；
- RAF/JPG、白平衡、标尺过滤和分割质量检查；
- 所有命令行参数和 `config.yaml` 字段；
- Python 公共函数、类、参数和返回值；
- U-Net 训练、颜色校准、输出字段和常见问题。

## Windows 最短运行流程

```powershell
cd leaf_color_phenotyping
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\batch_extract.py --no-aggregate --visualize --verbose
```

图片放入 `leaf_color_phenotyping/data/raw_images/`，结果生成在 `leaf_color_phenotyping/output/`。
