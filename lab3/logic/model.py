import json
import os

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from lab3.logic.config import Paths


class COCODataset(Dataset):
    def __init__(self, root, annotation_file):
        self.root = root
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)

        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = self.coco['annotations']

        self.img_to_anns = {}
        for ann in self.annotations:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)

        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train(
        train_loader,
        val_loader,
        num_epochs,
        patience,
):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 2  # background (0) + object (1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.005)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    patience_counter = 0

    print(f"Обучение запущено на устройстве: {device}")

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_train_loss += losses.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        epoch_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict_val = model(images, targets)
                losses_val = sum(loss for loss in loss_dict_val.values())
                epoch_val_loss += losses_val.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Paths.BEST_MODEL_DIST)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Ранняя остановка! Нет улучшений на валидации в течение {patience} эпох.")
                break
    return model, device, train_losses, val_losses



def draw_loss_epochs(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red', linestyle='--')
    plt.title('История обучения модели')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

