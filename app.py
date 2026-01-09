from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
import yaml
from detection_model import DishDetectionModel
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# 全局变量
model = None
config = None
dish_classes = []

def load_config(config_path: str = "config.yaml"):
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def init_model():
    """初始化模型"""
    global model, config, dish_classes
    
    config = load_config()
    dish_classes = config['data']['dish_classes']
    
    model_path = config['detection']['save_path']
    if Path(model_path).exists():
        model = DishDetectionModel(
            model_path=model_path,
            model_type=config['detection']['model_type'],
            num_classes=config['detection']['num_classes'],
            device=config['training']['device']
        )
        print("模型加载成功！")
    else:
        print(f"警告: 模型文件 {model_path} 不存在，请先训练模型")

def allowed_file(filename):
    """检查文件扩展名"""
    global config
    if config is None:
        config = load_config()
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config['api']['allowed_extensions']

def draw_detections(image_path: str, detections: list) -> str:
    """在图像上绘制检测结果"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        confidence = det['confidence']
        class_name = det['class_name']
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 保存结果图像
    output_path = Path('static') / 'results' / Path(image_path).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return str(output_path)

@app.route('/')
def index():
    """主页"""
    global config, dish_classes
    
    # 确保配置已加载
    if config is None or not dish_classes:
        try:
            config = load_config()
            dish_classes = config['data']['dish_classes']
        except Exception as e:
            print(f"加载配置失败: {e}")
            dish_classes = ["配置加载失败"]
    
    return render_template('index.html', dish_classes=dish_classes)

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'}), 400
    
    if model is None:
        return jsonify({
            'error': '模型未加载，请先训练模型',
            'message': '请参考 QUICKSTART.md 了解如何准备数据集和训练模型'
        }), 500
    
    # 保存上传的文件
    upload_dir = Path('static') / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = upload_dir / filename
    file.save(str(filepath))
    
    # 预测
    conf_threshold = config['detection']['confidence_threshold']
    iou_threshold = config['detection']['iou_threshold']
    
    result = model.predict(
        str(filepath),
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        save=False
    )
    
    # 绘制检测结果
    try:
        result_image_path = draw_detections(str(filepath), result['detections'])
        result_image_url = f'/static/results/{Path(result_image_path).name}'
    except Exception as e:
        result_image_url = None
        print(f"绘制检测结果时出错: {e}")
    
    # 准备响应
    response = {
        'success': True,
        'num_detections': result['num_detections'],
        'detections': result['detections'],
        'result_image': result_image_url
    }
    
    return jsonify(response)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """批量预测接口"""
    if 'files' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '文件列表为空'}), 400
    
    if model is None:
        return jsonify({
            'error': '模型未加载',
            'message': '请先训练模型。参考 QUICKSTART.md 了解详细步骤'
        }), 500
    
    # 保存文件
    upload_dir = Path('static') / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    filepaths = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = upload_dir / filename
            file.save(str(filepath))
            filepaths.append(str(filepath))
    
    if not filepaths:
        return jsonify({'error': '没有有效的文件'}), 400
    
    # 批量预测
    conf_threshold = config['detection']['confidence_threshold']
    iou_threshold = config['detection']['iou_threshold']
    
    results = model.predict_batch(
        filepaths,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    # 为每个结果绘制检测框
    for i, result in enumerate(results):
        if result['detections']:
            try:
                result_image_path = draw_detections(result['image_path'], result['detections'])
                result['result_image'] = f'/static/results/{Path(result_image_path).name}'
            except Exception as e:
                print(f"绘制检测结果时出错: {e}")
                result['result_image'] = None
    
    return jsonify({
        'success': True,
        'results': results
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """获取菜品类别列表"""
    global dish_classes
    if not dish_classes:
        try:
            config = load_config()
            dish_classes = config['data']['dish_classes']
        except:
            pass
    return jsonify({
        'classes': dish_classes,
        'num_classes': len(dish_classes)
    })

@app.route('/test')
def test_template():
    """测试模板渲染"""
    global config, dish_classes
    if config is None or not dish_classes:
        try:
            config = load_config()
            dish_classes = config['data']['dish_classes']
        except Exception as e:
            return f"配置加载失败: {e}", 500
    
    return f"配置已加载，菜品类别: {dish_classes}"

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # 确保配置在启动时就加载
    if config is None:
        init_model()
    
    print(f"菜品类别: {dish_classes}")
    print(f"启动Web服务: http://{config['api']['host'] if config else '0.0.0.0'}:{config['api']['port'] if config else 5000}")
    
    app.run(
        host=config['api']['host'] if config else '0.0.0.0',
        port=config['api']['port'] if config else 5000,
        debug=config['api']['debug'] if config else False
    )

