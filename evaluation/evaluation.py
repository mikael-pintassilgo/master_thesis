import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import os

def load_label_studio_boxes(json_path, image_shape, label_key='rectangle'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    boxes = []
    for item in data:
        for ann in item['annotations']:
            for result in ann['result']:
                if result['type'] == 'rectangle':
                    value = result['value']
                    x = int(value['x'] / 100 * image_shape[1])
                    y = int(value['y'] / 100 * image_shape[0])
                    w = int(value['width'] / 100 * image_shape[1])
                    h = int(value['height'] / 100 * image_shape[0])
                    box = [x, y, x + w, y + h]
                    boxes.append(box)
    return boxes

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_model(predictions, ground_truths, iou_threshold=0.5):
    y_true = []
    y_pred = []
    for gt_boxes, pred_boxes in zip(ground_truths, predictions):
        for gt_box in gt_boxes:
            y_true.append(1)
            matched = False
            for pred_box in pred_boxes:
                iou = calculate_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    matched = True
                    break
            y_pred.append(1 if matched else 0)
        for pred_box in pred_boxes:
            if not any(calculate_iou(pred_box, gt_box) >= iou_threshold for gt_box in gt_boxes):
                y_true.append(0)
                y_pred.append(1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1

def visualize_results(image, ground_truths, predictions):
    for box in ground_truths:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    for box in predictions:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# Example usage
image_path = 'path_to_image.jpg'
gt_json_path = 'ground_truth_labelstudio.json'
pred_json_path = 'prediction_labelstudio.json'

image = cv2.imread(image_path)
image_shape = image.shape[:2]  # (height, width)

ground_truths = [load_label_studio_boxes(gt_json_path, image_shape)]
predictions = [load_label_studio_boxes(pred_json_path, image_shape)]

precision, recall, f1 = evaluate_model(predictions, ground_truths)
print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

visualize_results(image, ground_truths[0], predictions[0])
