import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt, patches

import xml.etree.ElementTree as ET

from lab3.logic.config import Paths
from lab3.logic.extraction import ExtractionMethod


def get_extracted_boxes_from_json(json_path, frame_name):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    img_id = next(item['id'] for item in coco['images'] if item["file_name"] == frame_name)
    boxes = [ann['bbox'] for ann in coco['annotations'] if ann['image_id'] == img_id]
    return boxes


def compare_methods_visualize(frame_name):
    source_xml = Paths.SOURCE_ANNOTATIONS

    img_path = os.path.join(Paths.IMAGES_DIR, frame_name)

    base_img = cv2.imread(img_path)
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    source_boxes = []
    if os.path.exists(source_xml):
        tree = ET.parse(source_xml)
        root = tree.getroot()
        for tag in root.iter():
            if tag.tag == 'image' and frame_name in tag.get('name', ''):
                for box in tag.findall('box'):
                    source_boxes.append([int(float(box.get('xtl'))), int(float(box.get('ytl'))),
                                         int(float(box.get('xbr'))), int(float(box.get('ybr')))])
            elif tag.tag == 'track':
                for box in tag.findall('box'):
                    if int(box.get('frame')) == int(frame_name.split('_')[1].split('.')[0]) and box.get(
                            'outside') != '1':
                        source_boxes.append([int(float(box.get('xtl'))), int(float(box.get('ytl'))),
                                             int(float(box.get('xbr'))), int(float(box.get('ybr')))])


    fig, axes = plt.subplots(1, len(ExtractionMethod), figsize=(24, 8))

    for ax, (method_name, json_path) in zip(axes, {m.label: m.path for m in ExtractionMethod}.items()):
        img_draw = base_img.copy()

        # Рамки эталона
        for xtl, ytl, xbr, ybr in source_boxes:
            cv2.rectangle(img_draw, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)

        # Рамки извл.
        ext_boxes = get_extracted_boxes_from_json(json_path, frame_name)
        for x, y, w, h in ext_boxes:
            cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)  # Красный

        ax.imshow(img_draw)
        ax.set_title(method_name, fontsize=16)
        ax.axis('off')


    green_patch = patches.Patch(color='lime', label='Эталон')
    red_patch = patches.Patch(color='red', label='Извлечено')

    axes[0].legend(handles=[green_patch, red_patch], loc='lower right', fontsize=14,
                   facecolor='black', labelcolor='white', framealpha=0.7)

    plt.tight_layout()
    plt.show()


def why_hsv_is_bad_explanation(frame_num: int):
    cap = cv2.VideoCapture(Paths.SOURCE_OUTPUT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Пример зелёного
    lower_color = np.array([40, 50, 50])
    upper_color = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Оригинал")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Маска HSV (зел.)")
    plt.axis('off')
    plt.show()
