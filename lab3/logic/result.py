import cv2
import torch
import torchvision
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt, patches
from torchmetrics.detection import MeanAveragePrecision


def count_map(model, device, val_loader):
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    print("Расчет метрик на val...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            preds = model(images)

            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds_cpu, targets_cpu)

    map_results = metric.compute()
    print(f"mAP @ [IoU=0.50:0.95]: {map_results['map']:.4f}")
    print(f"mAP @ [IoU=0.50]:      {map_results['map_50']:.4f}")


def plot_predictions_row(indices, dataset, model, device, score_threshold=0.5):
    model.eval()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, idx in zip(axes, indices):
        image_tensor, target = dataset[idx]

        with torch.no_grad():
            pred = model([image_tensor.to(device)])[0]

        # [C, H, W] -> [H, W, C]
        image = image_tensor.permute(1, 2, 0).cpu().numpy()

        if image.min() < 0 or image.max() > 1:
            image = (image - image.min()) / (image.max() - image.min())

        ax.imshow(image)

        # Source
        for box in target['boxes']:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        # Predict
        boxes = pred['boxes'].cpu()
        scores = pred['scores'].cpu()

        # NMS для удаления дубликатов предиктов
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.3)
        boxes = boxes[keep].numpy()
        scores = scores[keep].numpy()

        for box, score in zip(boxes, scores):
            if score > score_threshold:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                # скор модели над рамкой
                ax.text(x1, y1 - 4, f"{score:.2f}", color='lime', fontsize=9,
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        ax.set_title(f"Индекс кадра: {idx}", fontsize=12)
        ax.axis('off')

    plt.suptitle("Сравнение разметки (Синий: Source, Зеленый: Prediction)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


def make_predicted_video(input_video_path, output_video_path, model, device, score_thresh=0.6):
    model.eval()
    cap = cv2.VideoCapture(input_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Видео создаётся...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_tensor = F.to_tensor(rgb_frame).to(device)

        with torch.no_grad():
            pred = model([img_tensor])[0]

        boxes = pred['boxes'].cpu()
        scores = pred['scores'].cpu()

        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.3)
        boxes = boxes[keep].numpy()
        scores = scores[keep].numpy()

        for box, score in zip(boxes, scores):
            if score > score_thresh:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Car {score:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Видео сохранено в файл: {output_video_path}")
