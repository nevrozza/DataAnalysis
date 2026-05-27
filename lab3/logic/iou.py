import json
import os
import xml.etree.ElementTree as ET


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)


def load_source_boxes(source_xml_path) -> dict:
    source_boxes = {}

    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    def add_box(img_k: str, box_tag):
        xtl = float(box_tag.get('xtl'))
        ytl = float(box_tag.get('ytl'))
        xbr = float(box_tag.get('xbr'))
        ybr = float(box_tag.get('ybr'))
        source_boxes.setdefault(img_k, []).append([xtl, ytl, xbr, ybr])

    for track_tag in root.findall('track'):
        for box_tag in track_tag.findall('box'):
            if box_tag.get('outside') == '1':
                continue
            frame_id = int(box_tag.get('frame'))
            add_box(f"frame_{frame_id:04d}.jpg", box_tag)

    for image_tag in root.findall('image'):
        img_name = os.path.basename(image_tag.get('name'))
        for box_tag in image_tag.findall('box'):
            add_box(img_name, box_tag)

    return source_boxes


def evaluate_json_file(json_path, source_boxes) -> tuple[float, int]:
    """Расчет среднего IOU относительно источника"""

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    extracted_boxes = {}

    for ann in coco_data['annotations']:
        img_name = img_id_to_name[ann['image_id']]
        x, y, w, h = ann['bbox']
        extracted_boxes.setdefault(img_name, []).append([x, y, x + w, y + h])

    total_iou = 0.0
    total_evaluated_boxes = 0

    for img_name, exts in extracted_boxes.items():
        sources = source_boxes.get(img_name, [])
        if not sources:
            continue

        for ext in exts:
            best_iou = max([calculate_iou(ext, source) for source in sources], default=0.0)
            total_iou += best_iou
            total_evaluated_boxes += 1

    if total_evaluated_boxes == 0:
        return 0.0, 0

    return total_iou / total_evaluated_boxes, total_evaluated_boxes
