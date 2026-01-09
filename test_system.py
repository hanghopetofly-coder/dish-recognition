"""
系统测试脚本
用于快速测试系统各个模块是否正常工作
"""

import sys
from pathlib import Path
import yaml

def test_imports():
    """测试所有必要的模块是否可以导入"""
    print("=" * 50)
    print("测试模块导入...")
    print("=" * 50)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False
    
    try:
        import ultralytics
        print(f"✓ Ultralytics {ultralytics.__version__}")
    except ImportError as e:
        print(f"✗ Ultralytics 导入失败: {e}")
        print("  请运行: pip install ultralytics")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV 导入失败: {e}")
        return False
    
    try:
        import flask
        print(f"✓ Flask {flask.__version__}")
    except ImportError as e:
        print(f"✗ Flask 导入失败: {e}")
        return False
    
    try:
        from data_preprocessing import prepare_yolo_dataset
        print("✓ data_preprocessing 模块")
    except ImportError as e:
        print(f"✗ data_preprocessing 导入失败: {e}")
        return False
    
    try:
        from detection_model import DishDetectionModel
        print("✓ detection_model 模块")
    except ImportError as e:
        print(f"✗ detection_model 导入失败: {e}")
        return False
    
    print("\n所有模块导入成功！\n")
    return True

def test_config():
    """测试配置文件"""
    print("=" * 50)
    print("测试配置文件...")
    print("=" * 50)
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("✗ config.yaml 文件不存在")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查必要的配置项
        required_keys = ['data', 'detection', 'training', 'evaluation', 'api']
        for key in required_keys:
            if key not in config:
                print(f"✗ 配置文件中缺少 '{key}' 配置项")
                return False
        
        print("✓ 配置文件格式正确")
        print(f"  - 菜品类别数: {len(config['data']['dish_classes'])}")
        print(f"  - 模型类型: {config['detection']['model_type']}")
        print(f"  - 训练设备: {config['training']['device']}")
        print("\n配置文件检查通过！\n")
        return True
    except Exception as e:
        print(f"✗ 配置文件读取失败: {e}")
        return False

def test_directories():
    """测试必要的目录结构"""
    print("=" * 50)
    print("测试目录结构...")
    print("=" * 50)
    
    directories = [
        'templates',
        'static',
        'static/uploads',
        'static/results',
        'models',
        'results',
        'data',
        'data/train',
        'data/val',
        'data/test'
    ]
    
    all_exist = True
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            print(f"✗ 目录不存在: {dir_path}")
            print(f"  正在创建...")
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ 已创建: {dir_path}")
        else:
            print(f"✓ {dir_path}")
    
    print("\n目录结构检查完成！\n")
    return True

def test_model_initialization():
    """测试模型初始化"""
    print("=" * 50)
    print("测试模型初始化...")
    print("=" * 50)
    
    try:
        from detection_model import DishDetectionModel
        
        # 尝试初始化模型（不加载权重）
        print("正在初始化模型（首次运行会下载预训练权重）...")
        model = DishDetectionModel(
            model_type="yolov8n",
            num_classes=10,
            device="cpu"  # 使用CPU避免CUDA问题
        )
        print("✓ 模型初始化成功")
        print("\n模型初始化测试通过！\n")
        return True
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        print("  这可能是正常的，如果模型文件不存在")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("食堂菜品检测与识别系统 - 系统测试")
    print("=" * 50 + "\n")
    
    results = []
    
    # 运行各项测试
    results.append(("模块导入", test_imports()))
    results.append(("配置文件", test_config()))
    results.append(("目录结构", test_directories()))
    results.append(("模型初始化", test_model_initialization()))
    
    # 汇总结果
    print("=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过！系统准备就绪。")
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
    print("=" * 50 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

