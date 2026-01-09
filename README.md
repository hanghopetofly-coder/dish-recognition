# 食堂菜品检测与识别系统

一个完整的基于深度学习的食堂菜品检测与识别系统，使用YOLOv8实现菜品的自动检测和分类识别。

## 功能特性

- 🎯 **目标检测**：使用YOLOv8模型检测图像中的菜品位置
- 🏷️ **分类识别**：自动识别检测到的菜品类别
- 🖼️ **数据预处理**：自动数据增强、归一化、YOLO格式数据加载
- 📊 **模型训练**：完整的训练流程，支持多种YOLOv8模型变体
- 📈 **模型评估**：mAP、精确率、召回率等目标检测指标
- 🌐 **Web服务**：RESTful API和Web界面，支持图像上传和实时预测
- 📦 **批量处理**：支持多张图像同时识别
- 🎨 **结果可视化**：自动绘制检测框和标签
- ⚙️ **配置管理**：YAML配置文件，易于调整超参数

## 项目结构

```
.
├── config.yaml              # 配置文件
├── requirements.txt         # Python依赖
├── data_preprocessing.py    # 数据预处理模块
├── detection_model.py       # 检测模型模块
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── app.py                   # Flask Web服务
├── templates/               # HTML模板
│   └── index.html
├── static/                  # 静态文件
│   ├── style.css
│   ├── uploads/            # 上传文件目录
│   └── results/            # 结果图像目录
├── models/                  # 保存的模型
├── results/                 # 评估结果
├── USAGE.md                 # 详细使用说明
└── data/                    # 数据目录
    ├── train/
    │   ├── images/         # 训练图像
    │   └── labels/         # YOLO格式标注
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## 安装步骤

1. **克隆或下载项目**

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备数据**
   - 按照YOLO格式组织数据（详见 `USAGE.md`）
   - 将训练图像放在 `data/train/images/` 目录
   - 将YOLO格式标注放在 `data/train/labels/` 目录
   - 同样方式组织验证集和测试集

4. **配置参数**
   - 编辑 `config.yaml` 文件，设置菜品类别、模型类型、训练参数等

## 使用方法

### 1. 训练模型

```bash
python train.py
```

### 2. 评估模型

```bash
python evaluate.py
```

### 3. 启动Web服务

```bash
python app.py
```

然后在浏览器中访问 `http://localhost:5000`

### 4. 使用API

```bash
# 预测单张图像
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

## 配置说明

主要配置项在 `config.yaml` 中：

- **data**: 数据路径、图像尺寸、批次大小、菜品类别列表
- **detection**: 检测模型类型（yolov8n/s/m/l/x）、类别数、置信度阈值
- **training**: 训练轮数、学习率、优化器参数、设备选择
- **evaluation**: 评估指标、结果保存设置
- **api**: Web服务端口和设置

## 技术栈

- **深度学习框架**: PyTorch
- **目标检测**: YOLOv8 (Ultralytics)
- **Web框架**: Flask
- **图像处理**: OpenCV, PIL
- **数据处理**: NumPy, scikit-learn

## 快速开始

### 第一步：安装依赖
```bash
pip install -r requirements.txt
```

### 第二步：准备数据集（必须）
1. **不知道如何获取数据集？** → 查看 [数据集获取指南.md](数据集获取指南.md)
2. **收集图片**：拍摄或收集食堂菜品图片
3. **标注数据**：使用LabelImg工具标注（参考 `labeling_guide.md`）
4. **检查数据**：
   ```bash
   python prepare_data.py --action create  # 创建目录结构
   python prepare_data.py --action check   # 检查数据
   ```

### 第三步：训练模型
```bash
python train.py
```

### 第四步：启动Web服务
```bash
python app.py
```

### 第五步：访问界面
打开浏览器访问: http://localhost:5000

**注意**：没有训练好的模型无法实际检测菜品，但可以查看界面。详细说明请参考 `QUICKSTART.md`

## 详细文档

更多详细信息请参考：
- **[数据集获取指南.md](数据集获取指南.md)** - 不知道去哪里找数据集？这里有完整的数据集获取方案
- **[USAGE.md](USAGE.md)** - 详细使用说明，包括：
  - 数据标注格式说明
  - 训练和评估详细步骤
  - API接口文档
  - 性能优化建议
  - 常见问题解答
- **[labeling_guide.md](labeling_guide.md)** - 数据标注详细指南

## 许可证

MIT License

