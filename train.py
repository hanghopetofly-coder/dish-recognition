import yaml
import argparse
from pathlib import Path
from detection_model import DishDetectionModel
from data_preprocessing import prepare_yolo_dataset

def load_config(config_path: str = "config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='训练食堂菜品检测模型')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 准备数据集
    data_yaml = prepare_yolo_dataset(
        config['data']['data_dir'],
        config['data']['dish_classes']
    )
    
    # 初始化模型
    model = DishDetectionModel(
        model_type=config['detection']['model_type'],
        num_classes=config['detection']['num_classes'],
        device=config['training']['device']
    )
    
    # 训练模型
    print("开始训练...")
    results = model.train(
        data_yaml=str(data_yaml),
        epochs=config['training']['epochs'],
        imgsz=config['data']['image_size'][0],
        batch=config['data']['batch_size'],
        lr0=config['training']['learning_rate'],
        device=config['training']['device']
    )
    
    # 保存模型
    model.save_model(config['detection']['save_path'])
    
    print("训练完成！")
    print(f"模型已保存到: {config['detection']['save_path']}")

if __name__ == '__main__':
    main()

