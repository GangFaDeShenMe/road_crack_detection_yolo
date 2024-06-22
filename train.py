from ultralytics import YOLO

if __name__=='__main__':
    # 配置文件路径
    config_path = './dataset/processed/1/dataset.yaml'

    # 模型路径
    model_path = 'yolov9c.pt'  # 你可以选择其他预训练模型，如 yolov8m.pt, yolov8l.pt 等

    # 训练参数
    epochs = 100  # 训练轮数
    batch = 16  # 批次大小
    imgsz = 640  # 输入图像大小
    device = '0'  # 使用第一个GPU

    # 创建并训练模型
    model = YOLO(model_path)
    model.train(data=config_path, epochs=epochs, batch=batch, imgsz=imgsz, device=device)

    # 保存训练好的模型
    model.save('best_model.pt')

