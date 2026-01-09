import yaml
import argparse
from pathlib import Path
from detection_model import DishDetectionModel
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_config(config_path: str = "config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def visualize_results(results: dict, save_path: str):
    """可视化评估结果"""
    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
    values = [results.get(m, 0) for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    plt.ylabel('分数', fontsize=12)
    plt.title('模型评估指标', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"评估结果图已保存到: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='评估食堂菜品检测模型')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径（覆盖配置文件）')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 准备数据集
    from data_preprocessing import prepare_yolo_dataset
    data_yaml = prepare_yolo_dataset(
        config['data']['data_dir'],
        config['data']['dish_classes']
    )
    
    # 加载模型
    model_path = args.model or config['detection']['save_path']
    model = DishDetectionModel(
        model_path=model_path,
        model_type=config['detection']['model_type'],
        num_classes=config['detection']['num_classes'],
        device=config['training']['device']
    )
    
    # 评估模型
    print("开始评估...")
    results = model.evaluate(str(data_yaml))
    
    # 打印结果
    print("\n=== 评估结果 ===")
    print(f"mAP@0.5: {results['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    
    # 保存结果
    if config['evaluation']['save_results']:
        results_dir = Path(config['evaluation']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON
        with open(results_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 可视化
        if config['evaluation']['visualize']:
            visualize_results(results, results_dir / 'evaluation_metrics.png')
    
    print("\n评估完成！")

if __name__ == '__main__':
    main()

