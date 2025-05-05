
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def convert_txt_to_labelstudio_json(txt_file, image_shape):
    boxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            if (len(line) <= 1):
                continue
            coords = list(map(int, line.strip().split(',')))
            x_coords = [coords[0], coords[2], coords[4], coords[6]]
            y_coords = [coords[1], coords[3], coords[5], coords[7]]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            width = x_max - x_min
            height = y_max - y_min
            # Convert to percentages for Label Studio
            x_perc = x_min / image_shape[1] * 100
            y_perc = y_min / image_shape[0] * 100
            w_perc = width / image_shape[1] * 100
            h_perc = height / image_shape[0] * 100
            box = {
                "original_width": image_shape[1],
                "original_height": image_shape[0],
                "image_rotation": 0,
                "value": {
                    "x": x_perc,
                    "y": y_perc,
                    "width": w_perc,
                    "height": h_perc,
                    "rotation": 0
                },
                "id": str(np.random.randint(1e9)),
                "from_name": "label",
                "to_name": "image",
                "type": "rectangle"
            }
            boxes.append(box)
    # JSON_MIN format for Label Studio
    result = [{
        "annotations": [{
            "result": boxes
        }]
    }]
    return result

results_dir = './result'
predictions_dir = 'evaluation/prediction'
os.makedirs(predictions_dir, exist_ok=True)

for txt_file in os.listdir(results_dir):
    if txt_file.endswith('.txt'):
        txt_path = os.path.join(results_dir, txt_file)
        image_filename = os.path.splitext(txt_file)[0] + '.jpg'
        image_path = os.path.join(results_dir, image_filename)
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                image_shape = img.size  # (width, height)
                image_shape = (img.height, img.width)  # (height, width)
        else:
            continue  # Skip if image file does not exist
        json_data = convert_txt_to_labelstudio_json(txt_path, image_shape)
        json_filename = os.path.splitext(txt_file)[0] + '.json'
        json_path = os.path.join(predictions_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            print(f"JSON file saved to: {json_path}")
        