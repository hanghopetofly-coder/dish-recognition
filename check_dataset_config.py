#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查数据集配置是否正确
"""
import yaml
from pathlib import Path

def check_dataset_config():
    """检查数据集配置"""
    print("=" * 60)
    print("检查数据集配置...")
    print("=" * 60)
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    data_dir = Path(data_config['data_dir'])
    
    print(f"\n数据目录: {data_dir}")
    print(f"数据目录是否存在: {data_dir.exists()}")
    
    # 检查各个目录
    splits = {
        '训练集': data_dir / 'train',
        '验证集': data_dir / 'valid',
        '测试集': data_dir / 'test'
    }
    
    print("\n检查数据集目录:")
    total_images = 0
    total_labels = 0
    
    for split_name, split_dir in splits.items():
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if split_dir.exists():
            # 统计图片和标注文件
            images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            labels = list(labels_dir.glob('*.txt'))
            
            print(f"\n{split_name}:")
            print(f"  目录: {split_dir}")
            print(f"  图片数量: {len(images)}")
            print(f"  标注数量: {len(labels)}")
            print(f"  状态: {'✓ 正常' if len(images) == len(labels) and len(images) > 0 else '✗ 有问题'}")
            
            total_images += len(images)
            total_labels += len(labels)
        else:
            print(f"\n{split_name}:")
            print(f"  目录: {split_dir}")
            print(f"  状态: ✗ 不存在")
    
    # 检查类别配置
    print("\n" + "=" * 60)
    print("类别配置:")
    print("=" * 60)
    dish_classes = data_config['dish_classes']
    print(f"类别数量: {len(dish_classes)}")
    print(f"类别列表: {dish_classes}")
    print(f"配置中的num_classes: {config['detection']['num_classes']}")
    
    if len(dish_classes) == config['detection']['num_classes']:
        print("✓ 类别数量匹配")
    else:
        print("✗ 类别数量不匹配！")
    
    # 检查data.yaml
    data_yaml = data_dir / 'data.yaml'
    print("\n" + "=" * 60)
    print("检查data.yaml:")
    print("=" * 60)
    if data_yaml.exists():
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_yaml_config = yaml.safe_load(f)
        print(f"✓ data.yaml存在")
        print(f"  类别数 (nc): {data_yaml_config.get('nc', 'N/A')}")
        print(f"  类别名 (names): {data_yaml_config.get('names', 'N/A')}")
        print(f"  验证集路径 (val): {data_yaml_config.get('val', 'N/A')}")
        
        # 检查是否使用valid目录
        if 'valid' in str(data_yaml_config.get('val', '')):
            print("  ✓ 使用valid目录（正确）")
        elif 'val' in str(data_yaml_config.get('val', '')):
            print("  ⚠ 使用val目录（但数据集中是valid）")
    else:
        print(f"✗ data.yaml不存在，需要运行数据准备脚本")
    
    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    print(f"总图片数: {total_images}")
    print(f"总标注数: {total_labels}")
    print(f"配置是否正确: {'✓' if total_images > 0 and total_images == total_labels and len(dish_classes) == config['detection']['num_classes'] else '✗'}")
    
    return total_images > 0 and total_images == total_labels

if __name__ == '__main__':
    check_dataset_config()

