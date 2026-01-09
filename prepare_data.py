"""
数据准备辅助脚本
帮助用户快速准备和检查数据集
"""

import os
import shutil
from pathlib import Path
import yaml
from collections import Counter

def load_config(config_path: str = "config.yaml"):
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_directory_structure(data_dir: str):
    """创建数据集目录结构"""
    data_path = Path(data_dir)
    
    directories = {
        'train': ['images', 'labels'],
        'val': ['images', 'labels'],
        'test': ['images', 'labels']
    }
    
    print("=" * 60)
    print("创建数据集目录结构...")
    print("=" * 60)
    
    for split, subdirs in directories.items():
        for subdir in subdirs:
            dir_path = data_path / split / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ 创建目录: {dir_path}")
    
    print("\n目录结构创建完成！\n")
    return data_path

def check_dataset(data_dir: str, dish_classes: list):
    """检查数据集完整性"""
    data_path = Path(data_dir)
    
    print("=" * 60)
    print("检查数据集...")
    print("=" * 60)
    
    issues = []
    stats = {}
    
    for split in ['train', 'val', 'test']:
        images_dir = data_path / split / 'images'
        labels_dir = data_path / split / 'labels'
        
        if not images_dir.exists():
            issues.append(f"✗ {split}/images 目录不存在")
            continue
        
        if not labels_dir.exists():
            issues.append(f"✗ {split}/labels 目录不存在")
            continue
        
        # 获取所有图片文件
        image_files = list(images_dir.glob("*.jpg")) + \
                     list(images_dir.glob("*.png")) + \
                     list(images_dir.glob("*.jpeg"))
        
        # 获取所有标注文件
        label_files = list(labels_dir.glob("*.txt"))
        
        # 检查图片和标注是否匹配
        image_names = {f.stem for f in image_files}
        label_names = {f.stem for f in label_files}
        
        missing_labels = image_names - label_names
        missing_images = label_names - image_names
        
        stats[split] = {
            'images': len(image_files),
            'labels': len(label_files),
            'matched': len(image_names & label_names),
            'missing_labels': len(missing_labels),
            'missing_images': len(missing_images)
        }
        
        if missing_labels:
            issues.append(f"✗ {split}: {len(missing_labels)} 张图片缺少标注文件")
            if len(missing_labels) <= 5:
                issues.append(f"  示例: {list(missing_labels)[:5]}")
        
        if missing_images:
            issues.append(f"✗ {split}: {len(missing_images)} 个标注文件缺少对应图片")
    
    # 打印统计信息
    print("\n数据集统计:")
    print("-" * 60)
    total_images = 0
    total_labels = 0
    
    for split, stat in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  图片数量: {stat['images']}")
        print(f"  标注数量: {stat['labels']}")
        print(f"  匹配数量: {stat['matched']}")
        if stat['missing_labels'] > 0:
            print(f"  ⚠ 缺少标注: {stat['missing_labels']}")
        if stat['missing_images'] > 0:
            print(f"  ⚠ 缺少图片: {stat['missing_images']}")
        
        total_images += stat['images']
        total_labels += stat['labels']
    
    print(f"\n总计:")
    print(f"  图片总数: {total_images}")
    print(f"  标注总数: {total_labels}")
    
    # 检查标注文件内容
    print("\n检查标注文件格式...")
    print("-" * 60)
    
    class_counts = Counter()
    invalid_files = []
    
    for split in ['train', 'val', 'test']:
        labels_dir = data_path / split / 'labels'
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(dish_classes):
                                class_counts[class_id] += 1
                            else:
                                invalid_files.append(f"{label_file.name}: 类别ID {class_id} 超出范围")
                        else:
                            invalid_files.append(f"{label_file.name}: 格式错误（应为5个值）")
            except Exception as e:
                invalid_files.append(f"{label_file.name}: {e}")
    
    if class_counts:
        print("\n类别分布:")
        for class_id, count in sorted(class_counts.items()):
            class_name = dish_classes[class_id] if class_id < len(dish_classes) else f"未知({class_id})"
            print(f"  {class_name}: {count} 个标注")
    
    if invalid_files:
        print(f"\n⚠ 发现 {len(invalid_files)} 个问题文件:")
        for issue in invalid_files[:10]:  # 只显示前10个
            print(f"  {issue}")
        if len(invalid_files) > 10:
            print(f"  ... 还有 {len(invalid_files) - 10} 个问题")
    
    # 打印问题汇总
    if issues:
        print("\n" + "=" * 60)
        print("发现的问题:")
        print("=" * 60)
        for issue in issues:
            print(issue)
        return False
    else:
        print("\n✓ 数据集检查通过！")
        return True

def split_dataset(source_dir: str, data_dir: str, train_ratio: float = 0.7, 
                 val_ratio: float = 0.2, test_ratio: float = 0.1):
    """自动划分数据集"""
    import random
    
    source_path = Path(source_dir)
    data_path = Path(data_dir)
    
    # 创建目录结构
    create_directory_structure(data_dir)
    
    # 获取所有图片
    image_files = list(source_path.glob("*.jpg")) + \
                 list(source_path.glob("*.png")) + \
                 list(source_path.glob("*.jpeg"))
    
    if not image_files:
        print(f"✗ 在 {source_dir} 中未找到图片文件")
        return False
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 计算划分点
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 划分数据集
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print("=" * 60)
    print("划分数据集...")
    print("=" * 60)
    print(f"训练集: {len(train_files)} 张 ({len(train_files)/total*100:.1f}%)")
    print(f"验证集: {len(val_files)} 张 ({len(val_files)/total*100:.1f}%)")
    print(f"测试集: {len(test_files)} 张 ({len(test_files)/total*100:.1f}%)")
    
    # 复制文件
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split, files in splits.items():
        images_dir = data_path / split / 'images'
        labels_dir = data_path / split / 'labels'
        
        for img_file in files:
            # 复制图片
            shutil.copy(img_file, images_dir / img_file.name)
            
            # 复制对应的标注文件（如果存在）
            label_file = source_path / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, labels_dir / label_file.name)
    
    print("\n✓ 数据集划分完成！")
    return True

def generate_sample_labels(data_dir: str, dish_classes: list):
    """为没有标注的图片生成示例标注文件（仅用于测试）"""
    data_path = Path(data_dir)
    
    print("=" * 60)
    print("生成示例标注文件（仅用于测试）...")
    print("=" * 60)
    print("⚠ 警告: 这些是随机生成的示例标注，不能用于实际训练！")
    print("⚠ 您需要使用LabelImg等工具进行真实标注。\n")
    
    import random
    
    for split in ['train', 'val', 'test']:
        images_dir = data_path / split / 'images'
        labels_dir = data_path / split / 'labels'
        
        if not images_dir.exists():
            continue
        
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(images_dir.glob("*.jpg")) + \
                     list(images_dir.glob("*.png")) + \
                     list(images_dir.glob("*.jpeg"))
        
        created = 0
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                # 生成一个随机示例标注（仅用于测试）
                class_id = random.randint(0, len(dish_classes) - 1)
                x_center = random.uniform(0.3, 0.7)
                y_center = random.uniform(0.3, 0.7)
                width = random.uniform(0.1, 0.3)
                height = random.uniform(0.1, 0.3)
                
                with open(label_file, 'w') as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                created += 1
        
        if created > 0:
            print(f"{split}: 为 {created} 张图片生成了示例标注")
    
    print("\n✓ 示例标注生成完成（仅用于测试格式）")
    print("⚠ 请使用LabelImg进行真实标注！")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='数据准备辅助工具')
    parser.add_argument('--action', type=str, 
                       choices=['create', 'check', 'split', 'sample'],
                       default='check',
                       help='操作类型: create(创建目录), check(检查数据), split(划分数据集), sample(生成示例标注)')
    parser.add_argument('--source', type=str, default=None,
                       help='源目录（用于split操作）')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='数据目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    dish_classes = config['data']['dish_classes']
    data_dir = args.data_dir or config['data']['data_dir']
    
    if args.action == 'create':
        create_directory_structure(data_dir)
    
    elif args.action == 'check':
        if not Path(data_dir).exists():
            print(f"✗ 数据目录不存在: {data_dir}")
            print("请先运行: python prepare_data.py --action create")
            return
        check_dataset(data_dir, dish_classes)
    
    elif args.action == 'split':
        if not args.source:
            print("✗ 请指定源目录: --source <目录路径>")
            return
        split_dataset(args.source, data_dir)
    
    elif args.action == 'sample':
        print("⚠ 警告: 这将生成随机示例标注，仅用于测试格式！")
        response = input("是否继续？(y/n): ")
        if response.lower() == 'y':
            generate_sample_labels(data_dir, dish_classes)
        else:
            print("已取消")

if __name__ == '__main__':
    main()

