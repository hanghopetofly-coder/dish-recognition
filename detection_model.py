from ultralytics import YOLO
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np

class DishDetectionModel:
    """食堂菜品检测模型"""
    
    def __init__(self, model_path: str = None, model_type: str = "yolov8n",
                 num_classes: int = 10, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.num_classes = num_classes
        
        if model_path and Path(model_path).exists():
            print(f"加载已训练模型: {model_path}")
            self.model = YOLO(model_path)
        else:
            print(f"初始化新模型: {model_type}")
            self.model = YOLO(f"{model_type}.pt")
            # 如果类别数不是默认的80，需要修改模型
            if num_classes != 80:
                # 这里需要根据实际情况调整
                pass
    
    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640,
              batch: int = 16, lr0: float = 0.01, device: str = None):
        """训练模型"""
        device = device or self.device
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            device=device,
            project='./runs/detect',
            name='dish_detection',
            exist_ok=True
        )
        
        return results
    
    def predict(self, image_path: str, conf_threshold: float = 0.5,
                iou_threshold: float = 0.45, save: bool = False) -> Dict:
        """预测单张图像"""
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            save=save,
            device=self.device
        )
        
        # 解析结果
        result = results[0]
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                detections.append({
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': result.names[class_ids[i]]
                })
        
        return {
            'image_path': image_path,
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def predict_batch(self, image_paths: List[str], conf_threshold: float = 0.5,
                     iou_threshold: float = 0.45) -> List[Dict]:
        """批量预测"""
        results = self.model.predict(
            source=image_paths,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device
        )
        
        predictions = []
        for i, result in enumerate(results):
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for j in range(len(boxes)):
                    detections.append({
                        'bbox': boxes[j].tolist(),
                        'confidence': float(confidences[j]),
                        'class_id': int(class_ids[j]),
                        'class_name': result.names[class_ids[j]]
                    })
            
            predictions.append({
                'image_path': image_paths[i],
                'detections': detections,
                'num_detections': len(detections)
            })
        
        return predictions
    
    def save_model(self, save_path: str):
        """保存模型"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # YOLO模型本身就是.pt格式，直接保存
        # 保存最佳模型（如果存在）
        best_model_path = Path('./runs/detect/dish_detection/weights/best.pt')
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, save_path)
            print(f"模型已保存到: {save_path}")
        else:
            # 如果没有训练过的模型，保存当前模型
            try:
                self.model.save(save_path)
                print(f"模型已保存到: {save_path}")
            except Exception as e:
                print(f"警告: 保存模型时出错: {e}")
                print(f"提示: 请先训练模型，训练后的模型会自动保存")
    
    def evaluate(self, data_yaml: str) -> Dict:
        """评估模型"""
        metrics = self.model.val(data=data_yaml, device=self.device)
        
        return {
            'mAP50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
            'mAP50-95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
            'precision': float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0.0,
            'recall': float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0.0
        }
