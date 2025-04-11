from coco import create_coco_dataloaders
from model import MINDObjectDetector
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_iou
from scipy.optimize import linear_sum_assignment


class SimpleMatcher:
    def __call__(self, pred_logits, pred_boxes, targets):
        indices = []
        for b in range(pred_logits.size(0)):
            tgt_boxes = targets[b]["boxes"]
            tgt_labels = targets[b]["labels"]
            out_prob = pred_logits[b].softmax(-1)  # [num_queries, num_classes]
            out_bbox = pred_boxes[b]              # [num_queries, 4]

            cost_cls = -out_prob[:, tgt_labels]   # Cross-entropy cost
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)  # L1 distance

            C = cost_bbox + cost_cls
            C = C.detach().cpu()
            i, j = linear_sum_assignment(C)
            indices.append((torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64)))
        return indices


class SimpleSetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = ["labels", "boxes"]

    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes]
        pred_boxes = outputs["pred_boxes"]    # [B, num_queries, 4]

        indices = self.matcher(pred_logits, pred_boxes, targets)
        loss_dict = {}

        # Compute classification loss
        loss_cls = 0
        loss_bbox = 0

        num_valid = 0
        for b, (idx_pred, idx_tgt) in enumerate(indices):
            if len(idx_pred) == 0 or len(idx_tgt) == 0:
                continue

            tgt_classes = targets[b]["labels"][idx_tgt]
            pred_classes = pred_logits[b][idx_pred]
            loss_cls += F.cross_entropy(pred_classes, tgt_classes)

            tgt_boxes = targets[b]["boxes"][idx_tgt]
            pred_bboxes = pred_boxes[b][idx_pred]
            loss_bbox += F.l1_loss(pred_bboxes, tgt_boxes)

            num_valid += 1

        num_valid = max(num_valid, 1)
        loss_dict["loss_ce"] = loss_cls / num_valid
        loss_dict["loss_bbox"] = loss_bbox / num_valid


        if "pred_obj" in outputs:
            pred_obj = outputs["pred_obj"]
            target_obj = torch.zeros_like(pred_obj)
            for b, (idx_pred, idx_tgt) in enumerate(indices):
                target_obj[b, idx_pred] = 1.0
            loss_obj = F.binary_cross_entropy_with_logits(pred_obj, target_obj)
            loss_dict["loss_obj"] = loss_obj
            
        total_loss = sum(loss_dict[k] * self.weight_dict.get(k, 1.0)
                         for k in loss_dict)
        return total_loss, loss_dict

train_loader, test_loader = create_coco_dataloaders()

model = MINDObjectDetector(
    input_size=224,
    num_heads=12,
    dynamic_categories=91,
    rank=32,
    mode="supervised",
    adaptable_moe=True,
    initial_vigilance=0.75,
    vigilance_increment=0.05,
    modality="object_detection",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model and loss to GPU
model = model.to(device)
criterion = SimpleSetCriterion(
    num_classes=91,
    matcher=SimpleMatcher(),
    weight_dict={
        "loss_ce": 1.0,
        "loss_bbox": 5.0,
        "loss_obj": 1.0,
    }
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
from tqdm import tqdm
for epoch in range(1):
    total_loss = 0
    model.train()
    train_loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False)

    for images, targets in train_loop:
        tensors, masks = images.decompose()

        tensors = tensors.to(device)
        masks = masks.to(device)
        for tgt in targets:
            tgt["boxes"] = tgt["boxes"].to(device)
            tgt["labels"] = tgt["labels"].to(device)
            tgt["size"] = tgt["size"].to(device)

        outputs = model(tensors)

        loss, loss_dict = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "model.pth")