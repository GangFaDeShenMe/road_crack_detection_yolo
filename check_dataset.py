import os
import cv2


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on the image.
    :param image: The image to draw on.
    :param bbox: The bounding box in YOLO format (x_center, y_center, width, height).
    :param color: The color of the bounding box.
    :param thickness: The thickness of the bounding box.
    """
    h, w, _ = image.shape
    x_center, y_center, box_width, box_height = bbox
    x_center, y_center, box_width, box_height = x_center * w, y_center * h, box_width * w, box_height * h

    x_min = int(x_center - box_width / 2)
    y_min = int(y_center - box_height / 2)
    x_max = int(x_center + box_width / 2)
    y_max = int(y_center + box_height / 2)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image


def visualize_yolo_dataset(image_dir, label_dir):
    """
    Visualizes the YOLO dataset.
    :param image_dir: Directory containing the images.
    :param label_dir: Directory containing the labels.
    """
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            image_file_jpg = label_file.replace('.txt', '.jpg')
            image_file_jpeg = label_file.replace('.txt', '.jpeg')

            image_path_jpg = os.path.join(image_dir, image_file_jpg)
            image_path_jpeg = os.path.join(image_dir, image_file_jpeg)

            if os.path.exists(image_path_jpg):
                image_path = image_path_jpg
            elif os.path.exists(image_path_jpeg):
                image_path = image_path_jpeg
            else:
                continue  # Skip if neither jpg nor jpeg image exists

            image = cv2.imread(image_path)
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    image = draw_bbox(image, bbox)

            cv2.imshow('Image with Bounding Boxes', image)
            cv2.waitKey(0)  # Press any key to close the window


if __name__ == '__main__':
    image_directory = 'D:/Projects/road_crack_detection_yolo/dataset/processed/1/images/val'
    label_directory = 'D:/Projects/road_crack_detection_yolo/dataset/processed/1/labels/val'
    visualize_yolo_dataset(image_directory, label_directory)
