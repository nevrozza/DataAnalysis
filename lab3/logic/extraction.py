import json
import os
from enum import Enum
import cv2
import numpy as np
from ultralytics import YOLO

from lab3.logic.config import Paths


class ExtractionMethod(Enum):
    DIFF = ("Diff", f"{Paths.ANNOTATIONS_DIR}/diff.json",)
    HSV = ("HSV", f"{Paths.ANNOTATIONS_DIR}/hsv.json")
    MULTI_HSV = ("MULTI_HSV", f"{Paths.ANNOTATIONS_DIR}/multi_hsv.json")
    YOLO = ("YOLOv8", f"{Paths.ANNOTATIONS_DIR}/yolo.json")

    def __init__(self, label: str, path: str):
        self.label = label
        self.path = path


# ======= СТРАТЕГИИ ПОИСКА (per 1 frame) =======
def _find_boxes_diff(frame_in, frame_out, kernel) -> list:
    """ABSDIFF"""
    height, width, _ = frame_in.shape
    margin = 15
    rects = []

    diff = cv2.absdiff(frame_in, frame_out)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Игнорим рамки у края – так больше IOU
        is_touching_edge = (x < margin) or (y < margin) or (x + w > width - margin) or (y + h > height - margin)

        if 40 < w < width * 0.8 and 40 < h < height * 0.8 and not is_touching_edge:
            x_new, y_new, w_new, h_new = x + 2, y + 22, w - 24, h - 24
            if w_new > 10 and h_new > 10:
                rects.extend([[x_new, y_new, w_new, h_new], [x_new, y_new, w_new, h_new]])

    rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.15)
    return [list(r) for r in rects]


def _find_boxes_hsv(frame_out, kernel, lower_color, upper_color) -> list:
    """HSV по одному цвету"""
    height, width, _ = frame_out.shape
    margin = 15
    rects = []

    hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        is_touching_edge = (x < margin) or (y < margin) or (x + w > width - margin) or (y + h > height - margin)

        if 40 < w < width * 0.8 and 40 < h < height * 0.8 and not is_touching_edge:
            x_new, y_new, w_new, h_new = x + 2, y + 2, w - 4, h - 4
            rects.extend([[x_new, y_new, w_new, h_new], [x_new, y_new, w_new, h_new]])

    rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.1)
    return [list(r) for r in rects]


def _find_boxes_multi_hsv(frame_out, kernel, color_ranges: list) -> list:
    """HSV по нескольким цветам"""
    height, width, _ = frame_out.shape
    margin = 15
    rects = []

    hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV)

    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Накладываем маски всех цветов друг на друга
    for lower, upper in color_ranges:
        color_mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, color_mask)

    # Дальше логика как у обычного HSV
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        is_touching_edge = (x < margin) or (y < margin) or (x + w > width - margin) or (y + h > height - margin)

        if 40 < w < width * 0.8 and 40 < h < height * 0.8 and not is_touching_edge:
            x_new, y_new, w_new, h_new = x + 2, y + 2, w - 4, h - 4
            rects.extend([[x_new, y_new, w_new, h_new], [x_new, y_new, w_new, h_new]])

    rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.1)
    return [list(r) for r in rects]


def _find_boxes_yolo(frame_in, model) -> list:
    """YOLO"""
    rects = []
    results = model(frame_in, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id in [2, 7] and conf > 0.4:  # car, truck
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            rects.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    return rects


# ============= ПАЙПЛАН воровства =============

def run_extraction(method: ExtractionMethod, input_video: str = None, output_video: str = None,
                   save_frames: bool = False):
    cap_in = cv2.VideoCapture(input_video) if input_video else None
    cap_out = cv2.VideoCapture(output_video) if output_video else None

    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) if method == ExtractionMethod.DIFF else None
    kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) if method in (ExtractionMethod.HSV,
                                                                                 ExtractionMethod.MULTI_HSV) else None
    yolo_model = YOLO(Paths.YOLO_DIST) if method == ExtractionMethod.YOLO else None

    frame_id = 0
    ann_id = 0
    annotations = []
    images_info = []

    if save_frames:
        os.makedirs(Paths.IMAGES_DIR, exist_ok=True)

    while True:
        ret_in = cap_in.read() if cap_in else (False, None)
        ret_out = cap_out.read() if cap_out else (False, None)

        if (cap_in and not ret_in[0]) or (cap_out and not ret_out[0]):
            break

        frame_in, frame_out = ret_in[1], ret_out[1]
        base_frame = frame_in if frame_in is not None else frame_out

        image_name = f"frame_{frame_id:04d}.jpg"
        height, width, _ = base_frame.shape
        images_info.append({"id": frame_id, "file_name": image_name, "width": width, "height": height})

        if save_frames and frame_in is not None:
            cv2.imwrite(os.path.join(Paths.IMAGES_DIR, image_name), frame_in)

        if method == ExtractionMethod.DIFF:
            rects = _find_boxes_diff(frame_in, frame_out, kernel_3x3)
        elif method == ExtractionMethod.HSV:
            # green
            rects = _find_boxes_hsv(frame_out, kernel_5x5, lower_color=np.array([40, 50, 50]),
                                    upper_color=np.array([90, 255, 255]))
        elif method == ExtractionMethod.MULTI_HSV:
            rects = _find_boxes_multi_hsv(frame_out, kernel_5x5, color_ranges=multi_hsv_color_ranges)
        elif method == ExtractionMethod.YOLO:
            rects = _find_boxes_yolo(frame_in, yolo_model)
        else:
            rects = []

        for x, y, w, h in rects:
            annotations.append({
                "id": ann_id,
                "image_id": frame_id,
                "category_id": 1,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(w * h),
                "iscrowd": 0
            })
            ann_id += 1

        frame_id += 1

    if cap_in: cap_in.release()
    if cap_out: cap_out.release()

    coco_format = {"images": images_info, "annotations": annotations, "categories": [{"id": 1, "name": "object"}]}

    os.makedirs(os.path.dirname(method.path), exist_ok=True)
    with open(method.path, 'w') as f:
        json.dump(coco_format, f)

    print(f"[{method.label}] Извлечение завершено. Найдено рамок: {ann_id}")


multi_hsv_color_ranges = [
    (np.array([150, 100, 100]), np.array([170, 255, 255])),
    (np.array([92, 115, 175]), np.array([112, 215, 255])),
    (np.array([21, 15, 80]), np.array([41, 65, 180])),
    (np.array([103, 150, 40]), np.array([123, 255, 154])),
    (np.array([50, 100, 100]), np.array([70, 255, 255])),
    (np.array([11, 70, 180]), np.array([31, 170, 255])),
    (np.array([160, 110, 100]), np.array([179, 210, 200])),
    (np.array([21, 130, 190]), np.array([31, 230, 255])),
    (np.array([160, 65, 160]), np.array([175, 165, 255])),
    (np.array([60, 70, 180]), np.array([80, 170, 255])),
    (np.array([132, 120, 170]), np.array([152, 220, 255])),
    (np.array([90, 110, 190]), np.array([110, 210, 255])),
    (np.array([160, 20, 200]), np.array([179, 65, 255])),
    (np.array([75, 80, 50]), np.array([95, 180, 150])),
    (np.array([43, 165, 30]), np.array([63, 255, 130])),
    (np.array([114, 45, 185]), np.array([134, 125, 255])),
    (np.array([16, 90, 130]), np.array([30, 190, 240])),
    (np.array([145, 140, 130]), np.array([165, 240, 235])),
    (np.array([138, 40, 170]), np.array([158, 110, 255])),
    (np.array([40, 80, 100]), np.array([110, 255, 255])),
    (np.array([104, 180, 30]), np.array([124, 255, 110]))
]
