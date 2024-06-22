import os
import shutil
import random
import yaml

def create_yolo_dataset(input_dir, output_dir, class_names, train_ratio=0.8):
    # 创建输出目录结构
    image_train_dir = os.path.join(output_dir, 'images', 'train')
    image_val_dir = os.path.join(output_dir, 'images', 'val')
    label_train_dir = os.path.join(output_dir, 'labels', 'train')
    label_val_dir = os.path.join(output_dir, 'labels', 'val')

    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    # 获取所有标签文件
    label_files = [f for f in os.listdir(input_dir) if f.endswith('.txt') and f != 'classes.txt']
    random.shuffle(label_files)

    # 按照比例分割训练集和验证集
    train_count = int(len(label_files) * train_ratio)
    train_files = label_files[:train_count]
    val_files = label_files[train_count:]

    def process_files(files, image_output_dir, label_output_dir):
        for filename in files:
            input_txt_path = os.path.join(input_dir, filename)
            output_txt_path = os.path.join(label_output_dir, filename)

            # 直接复制标签文件
            shutil.copy(input_txt_path, output_txt_path)

            # 复制对应的图像文件到输出目录
            image_filename = filename.replace('.txt', '.jpeg')
            input_image_path = os.path.join(input_dir, image_filename)
            output_image_path = os.path.join(image_output_dir, image_filename)

            if not os.path.exists(input_image_path):
                image_filename = filename.replace('.txt', '.jpg')
                input_image_path = os.path.join(input_dir, image_filename)

            if os.path.exists(input_image_path):
                shutil.copy(input_image_path, output_image_path)
            else:
                print(f"Image file {image_filename} not found.")

    # 处理训练集和验证集文件
    process_files(train_files, image_train_dir, label_train_dir)
    process_files(val_files, image_val_dir, label_val_dir)

    # 生成配置文件
    config = {
        'train': os.path.join(output_dir, 'images', 'train'),
        'val': os.path.join(output_dir, 'images', 'val'),
        'nc': len(class_names),
        'names': class_names
    }
    config_file_path = os.path.join(output_dir, 'dataset.yaml')
    with open(config_file_path, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    print("Conversion completed, output files are saved in 'dataset/processed/1' directory.")

def create_yolo_dataset2(input_dir, output_dir, class_names, train_ratio=0.8):
    import xml.etree.ElementTree as ET

    # 创建输出目录结构
    image_train_dir = os.path.join(output_dir, 'images', 'train')
    image_val_dir = os.path.join(output_dir, 'images', 'val')
    label_train_dir = os.path.join(output_dir, 'labels', 'train')
    label_val_dir = os.path.join(output_dir, 'labels', 'val')

    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    # 记录每个类别最早出现的图片
    class_first_occurrence = {}

    # 获取所有XML文件
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    random.shuffle(xml_files)

    # 按比例分割训练集和验证集
    train_count = int(len(xml_files) * train_ratio)
    train_files = xml_files[:train_count]
    val_files = xml_files[train_count:]

    def convert_xml_to_yolo(xml_file, output_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)

        with open(output_file, 'w') as outfile:
            for obj in root.findall('object'):
                class_name = obj.find('name').text

                # 将类别 `potholes` 合并到 `pothole`
                if class_name == 'potholes':
                    class_name = 'pothole'

                # 忽略无效类别
                if class_name not in class_names:
                    continue

                class_id = class_names.index(class_name)

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                x_center = (xmin + xmax) / 2 / image_width
                y_center = (ymin + ymax) / 2 / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                outfile.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

    def process_files(files, image_output_dir, label_output_dir):
        for filename in files:
            input_xml_path = os.path.join(input_dir, filename)
            batch_name = '_'.join(filename.split('_')[:5])
            new_filename = f"{batch_name}.jpeg"
            output_txt_path = os.path.join(label_output_dir, new_filename.replace('.jpeg', '.txt'))

            convert_xml_to_yolo(input_xml_path, output_txt_path)

            # 复制对应的图像文件到输出目录
            input_image_path = os.path.join(input_dir, filename.replace('.xml', '.jpeg'))
            output_image_path = os.path.join(image_output_dir, new_filename)

            if not os.path.exists(input_image_path):
                input_image_path = os.path.join(input_dir, filename.replace('.xml', '.jpg'))
                output_image_path = os.path.join(image_output_dir, new_filename.replace('.jpeg', '.jpg'))

            if os.path.exists(input_image_path):
                shutil.copy(input_image_path, output_image_path)
            else:
                print(f"Image file {input_image_path} not found.")

    # 处理训练集和验证集文件
    process_files(train_files, image_train_dir, label_train_dir)
    process_files(val_files, image_val_dir, label_val_dir)

    # 输出类别最早出现的图片记录文件
    record_file_path = os.path.join(output_dir, 'class_first_occurrence.txt')
    with open(record_file_path, 'w') as record_file:
        for class_id, image_filename in sorted(class_first_occurrence.items()):
            record_file.write(f'Class {class_id} first appears in {image_filename}\n')

    # 更新配置文件
    config_file_path = os.path.join(output_dir, 'dataset.yaml')
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    config['nc'] = len(class_names)
    config['names'] = class_names

    with open(config_file_path, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    print("New dataset processed and merged, output files are saved in 'dataset/processed/1' directory.")

# 定义类别名称
class_names = ['pothole', 'alligator cracking', 'lateral cracking', 'longitudinal cracking']

# 输入和输出目录
input_dir1 = './dataset/processing/1'
input_dir2 = './dataset/processing/2/dataset/potholes'
output_dir = './dataset/processed/1'

# 创建YOLO数据集
create_yolo_dataset(input_dir1, output_dir, class_names)

# 合并新的数据集
create_yolo_dataset2(input_dir2, output_dir, class_names)
