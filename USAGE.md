# 食堂菜品检测与识别系统使用说明

## 系统简介

本系统是一个基于深度学习的食堂菜品检测与识别系统，使用YOLOv8模型实现菜品的自动检测和分类识别。系统支持Web界面和RESTful API两种使用方式。

## 功能特性

- 🎯 **目标检测**: 使用YOLOv8模型检测图像中的菜品位置
- 🏷️ **分类识别**: 自动识别检测到的菜品类别
- 🌐 **Web界面**: 友好的可视化界面，支持拖拽上传
- 🔌 **RESTful API**: 支持集成到其他系统
- 📊 **批量处理**: 支持多张图像同时识别
- 📈 **结果可视化**: 自动绘制检测框和标签

## 安装步骤

### 1. 环境要求

- Python 3.8+
- CUDA（可选，用于GPU加速）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 目录结构

确保项目目录结构如下：

```
.
├── config.yaml              # 配置文件
├── requirements.txt         # Python依赖
├── data_preprocessing.py    # 数据预处理模块
├── detection_model.py       # 检测模型模块
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── app.py                   # Flask Web服务
├── templates/              # HTML模板
│   └── index.html
├── static/                 # 静态文件
│   ├── style.css
│   ├── uploads/            # 上传文件目录
│   └── results/            # 结果图像目录
├── models/                 # 保存的模型
├── results/                # 评估结果
└── data/                   # 数据目录
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## 数据准备

### 1. 数据集结构

数据集需要按照YOLO格式组织：

```
data/
├── train/
│   ├── images/     # 训练图像（.jpg格式）
│   └── labels/     # YOLO格式标注文件（.txt格式）
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 2. 标注格式（YOLO格式）

每个标注文件（.txt）对应一张图像，文件名需一致（如：image001.jpg 对应 image001.txt）。

标注文件格式：
```
class_id x_center y_center width height
```

参数说明：
- `class_id`: 类别ID（从0开始，对应config.yaml中的dish_classes列表）
- `x_center, y_center`: 边界框中心点坐标（归一化到0-1）
- `width, height`: 边界框宽度和高度（归一化到0-1）

示例（image001.txt）：
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

这表示：
- 第一个菜品：类别0，中心在(0.5, 0.5)，宽高为(0.3, 0.4)
- 第二个菜品：类别1，中心在(0.2, 0.3)，宽高为(0.15, 0.2)

### 3. 标注工具推荐

- **LabelImg**: https://github.com/tzutalin/labelImg
  - 支持YOLO格式导出
  - 简单易用，适合快速标注
  
- **CVAT**: https://github.com/openvinotoolkit/cvat
  - 功能强大，支持团队协作
  - 适合大规模数据集标注

- **Roboflow**: https://roboflow.com/
  - 在线标注平台
  - 提供数据增强和版本管理

### 4. 数据准备脚本

运行以下命令自动创建数据集目录结构：

```python
from data_preprocessing import prepare_yolo_dataset
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

prepare_yolo_dataset(
    config['data']['data_dir'],
    config['data']['dish_classes']
)
```

## 训练流程

### 1. 配置参数

编辑 `config.yaml` 文件，设置以下参数：

- **数据配置**: 数据路径、图像尺寸、批次大小
- **模型配置**: 模型类型（yolov8n/s/m/l/x）、类别数
- **训练配置**: 训练轮数、学习率、设备（cuda/cpu）

### 2. 开始训练

```bash
python train.py
```

或者指定配置文件：

```bash
python train.py --config config.yaml
```

训练过程会显示：
- 训练进度
- 损失值变化
- 验证指标

训练完成后，模型会保存到 `./models/detection_model.pt`

### 3. 训练输出

训练过程中会在 `./runs/detect/dish_detection/` 目录下生成：
- `weights/best.pt`: 最佳模型权重
- `weights/last.pt`: 最后一个epoch的权重
- `results.png`: 训练曲线图
- `confusion_matrix.png`: 混淆矩阵

## 模型评估

### 1. 评估模型

```bash
python evaluate.py
```

或者指定模型路径：

```bash
python evaluate.py --model ./models/detection_model.pt
```

### 2. 评估指标

系统会输出以下指标：
- **mAP@0.5**: 在IoU=0.5时的平均精度
- **mAP@0.5:0.95**: 在IoU=0.5到0.95的平均精度
- **精确率 (Precision)**: 检测为正例中真正为正例的比例
- **召回率 (Recall)**: 真正例中被正确检测的比例

### 3. 评估结果

评估结果会保存到 `./results/` 目录：
- `evaluation_results.json`: JSON格式的评估结果
- `evaluation_metrics.png`: 可视化图表

## Web服务使用

### 1. 启动服务

```bash
python app.py
```

服务启动后，访问: http://localhost:5000

### 2. Web界面功能

- **上传图片**: 点击或拖拽图片到上传区域
- **批量上传**: 支持同时上传多张图片
- **实时识别**: 点击"开始识别"按钮进行识别
- **结果展示**: 显示检测结果图像和详细信息

### 3. API接口

#### 单张图像预测

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

响应示例：
```json
{
  "success": true,
  "num_detections": 2,
  "detections": [
    {
      "bbox": [100, 150, 300, 400],
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "红烧肉"
    }
  ],
  "result_image": "/static/results/image.jpg"
}
```

#### 批量预测

```bash
curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" http://localhost:5000/predict_batch
```

#### 获取类别列表

```bash
curl http://localhost:5000/classes
```

响应示例：
```json
{
  "classes": ["红烧肉", "宫保鸡丁", ...],
  "num_classes": 10
}
```

## 性能优化建议

### 1. 模型选择

- **yolov8n**: 最快，适合实时应用，精度较低
- **yolov8s**: 平衡速度和精度，推荐用于大多数场景
- **yolov8m**: 较高精度，适合对精度要求高的场景
- **yolov8l/x**: 最高精度，但速度较慢

### 2. 数据增强

- 增加训练数据的多样性
- 使用不同光照、角度、背景的图像
- 适当的数据增强可以提高模型泛化能力

### 3. 超参数调优

- **学习率**: 从0.01开始，根据训练情况调整
- **批次大小**: 根据GPU内存调整（16-32推荐）
- **训练轮数**: 观察验证集指标，避免过拟合

### 4. 硬件优化

- 使用GPU加速训练（CUDA）
- 增加数据加载的num_workers数量
- 使用混合精度训练（FP16）

## 常见问题

### 1. 模型加载失败

**问题**: 提示模型文件不存在

**解决**: 
- 确保已经完成模型训练
- 检查 `config.yaml` 中的 `save_path` 路径是否正确
- 如果使用预训练模型，确保网络连接正常（首次运行会下载模型）

### 2. 内存不足

**问题**: 训练时出现CUDA out of memory错误

**解决**:
- 减小 `batch_size`
- 减小 `image_size`
- 使用更小的模型（如yolov8n）

### 3. 检测精度低

**问题**: 模型检测精度不理想

**解决**:
- 增加训练数据量
- 检查标注质量
- 使用更大的模型
- 调整置信度阈值
- 增加训练轮数

### 4. 识别速度慢

**问题**: 预测速度太慢

**解决**:
- 使用更小的模型（yolov8n）
- 减小输入图像尺寸
- 使用GPU加速
- 考虑模型量化或剪枝

## 扩展功能

### 1. 添加新菜品类别

1. 在 `config.yaml` 的 `dish_classes` 中添加新类别
2. 更新 `num_classes` 参数
3. 重新标注数据（类别ID从0开始）
4. 重新训练模型

### 2. 集成到其他系统

系统提供RESTful API，可以轻松集成到：
- 食堂管理系统
- 点餐系统
- 营养分析系统
- 移动应用

### 3. 功能扩展建议

- **价格识别**: 结合OCR识别菜品价格
- **营养信息**: 添加菜品营养成分信息
- **推荐系统**: 基于历史数据推荐菜品
- **统计分析**: 菜品销量和偏好分析

## 技术支持

如有问题，请检查：
1. 依赖包是否正确安装
2. 配置文件格式是否正确
3. 数据格式是否符合要求
4. 模型文件是否存在

## 许可证

MIT License

