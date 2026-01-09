import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from ultralytics import YOLO
import json

class DishDataset(Dataset):
    """食堂菜品数据集"""
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (640, 640), 
                 mode: str = 'train', augment: bool = True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.mode = mode
        self.augment = augment and mode == 'train'
        
        # 加载图像和标注
        self.images = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """加载数据集"""
        image_dir = self.data_dir / self.mode / "images"
        label_dir = self.data_dir / self.mode / "labels"
        
        if not image_dir.exists():
            print(f"警告: {image_dir} 不存在，创建空数据集")
            return
        
        for img_file in image_dir.glob("*.jpg"):
            label_file = label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                self.images.append(str(img_file))
                self.labels.append(str(label_file))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]
        
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取YOLO格式标注
        boxes = []
        classes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append([x_center, y_center, width, height])
                    classes.append(class_id)
        
        # 数据增强
        if self.augment:
            image, boxes = self._augment(image, boxes)
        
        # 调整图像大小
        h, w = image.shape[:2]
        image = cv2.resize(image, self.image_size)
        
        # 转换为tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'classes': torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long)
        }
    
    def _augment(self, image, boxes):
        """数据增强"""
        # 水平翻转
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            boxes = [[1 - b[0], b[1], b[2], b[3]] for b in boxes]
        
        # 亮度调整
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        # 对比度调整
        if np.random.random() > 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            image = cv2.LUT(image, table)
        
        return image, boxes


def create_dataloader(data_dir: str, mode: str = 'train', 
                     batch_size: int = 16, image_size: Tuple[int, int] = (640, 640),
                     num_workers: int = 4) -> DataLoader:
    """创建数据加载器"""
    dataset = DishDataset(data_dir, image_size, mode, augment=(mode == 'train'))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return dataloader


def collate_fn(batch):
    """自定义collate函数"""
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    classes = [item['classes'] for item in batch]
    return {
        'images': images,
        'boxes': boxes,
        'classes': classes
    }


def prepare_yolo_dataset(data_dir: str, dish_classes: List[str]):
    """准备YOLO格式数据集"""
    data_path = Path(data_dir)
    
    # 检测验证集目录名（valid或val）
    val_dir_name = 'valid' if (data_path / 'valid').exists() else 'val'
    
    # 创建目录结构（如果不存在）
    for split in ['train', val_dir_name, 'test']:
        (data_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 创建data.yaml文件（YOLO格式）
    yolo_config = {
        'path': str(data_path.absolute()),
        'train': 'train/images',
        'val': f'{val_dir_name}/images',  # 使用实际存在的验证集目录名
        'test': 'test/images',
        'nc': len(dish_classes),
        'names': dish_classes
    }
    
    with open(data_path / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yolo_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"YOLO数据集配置已创建: {data_path / 'data.yaml'}")
    print(f"  使用验证集目录: {val_dir_name}")
    return data_path / 'data.yaml'
