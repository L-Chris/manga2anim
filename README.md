# Manga Panel Parser

> [!IMPORTANT]
> 🤖 本项目由 **Qwen3.8** 构建。

本地运行的漫画分镜解析桌面工具。导入漫画图片后，自动识别分镜、气泡和文字，重建阅读顺序，并允许手动修正后导出 JSON。

## 功能

- 批量导入漫画图片，保留每一页的解析结果
- 自动检测分镜、文字和气泡
- 支持从右到左、从左到右和竖向阅读顺序
- 本地 OCR，针对漫画竖排文字进行了优化
- 分镜边框、序号和文本流可视化
- 可编辑文字类型、所属分镜和分镜顺序
- 导出 `manga-panel-parser/v1` 格式 JSON
- 模型完全在本机运行，不上传漫画图片

## 处理流程

```text
图片导入
  → YOLO26s 检测分镜 / 文字 / 气泡
  → PP-OCRv6 识别文字
  → 重建阅读顺序并归属文本
  → 手动校正
  → JSON 导出
```

## 安装

Windows 用户可直接从 [Releases](https://github.com/L-Chris/manga2anim/releases/latest) 下载 `.exe` 或 `.msi` 安装包。

发布版已内置以下模型，安装后无需额外下载：

- YOLO26s Manga 实例分割模型
- PP-OCRv6 small 文本检测模型
- PP-OCRv6 medium 文字识别模型

## 本地开发

需要 Node.js 20+、Rust stable，以及 [Tauri 系统依赖](https://tauri.app/start/prerequisites/)。

```bash
npm install
npm run tauri dev
```

常用命令：

```bash
npm test             # 运行测试
npm run build        # TypeScript 检查和前端构建
npm run tauri build  # 构建桌面安装包
```

## 源码运行所需模型

自行运行源码时，需要在 `models/` 中准备：

| 文件 | 用途 |
| --- | --- |
| `yolo26s-manga-seg.onnx` | 分镜、文字和气泡检测 |
| `ppocrv6-det.onnx` | OCR 文本检测 |
| `ppocrv6-rec.onnx` | OCR 文字识别 |
| `ppocrv6_dict.txt` | PP-OCRv6 字符字典 |

YOLO 原始权重来自 [Manga109 YOLO26 segmentation](https://huggingface.co/ShadowB/Manga109-panel-balloon-text-yolov26-segmentation)，可运行以下命令导出 ONNX：

```bash
pip install ultralytics onnx
python scripts/export_onnx.py
```

## 技术栈

- Tauri 2、Rust
- React、TypeScript、Tailwind CSS
- ONNX Runtime Web
- YOLO26s、PP-OCRv6

## 数据格式

导出的 `manga-panel-parser/v1` 文档包含页面、分镜、阅读顺序、文本区域、文本类型及其所属分镜。未归属任何分镜的文字仅出现在 `otherTexts` 中。
