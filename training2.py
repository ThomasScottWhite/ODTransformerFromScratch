from coco import create_coco_dataloaders
from model import MINDObjectDetector
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_iou
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


class SimpleMatcher:
    def __call__(self, pred_logits, pred_boxes, targets):
        indices = []
        for b in range(pred_logits.size(0)):
            tgt_boxes = targets[b]["boxes"]
            tgt_labels = targets[b]["labels"]
            out_prob = pred_logits[b].softmax(-1)  # [num_queries, num_classes]
            out_bbox = pred_boxes[b]  # [num_queries, 4]

            cost_cls = -out_prob[:, tgt_labels]  # Cross-entropy cost
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)  # L1 distance

            C = cost_bbox + cost_cls
            C = C.detach().cpu()
            i, j = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
            )
        return indices


import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou


class HungarianMatcher:
    """
    Almost‑drop‑in replacement for SimpleMatcher.
    λ_class, λ_bbox, λ_giou follow DETR (1, 5, 2) by default.
    """

    def __init__(self, lambda_cls=1.0, lambda_bbox=5.0, lambda_giou=2.0):
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou

    @torch.no_grad()
    def __call__(self, pred_logits, pred_boxes, targets):
        bs, num_queries, _ = pred_logits.shape
        device = pred_logits.device

        indices = []
        for b in range(bs):
            tgt_boxes = targets[b]["boxes"]
            tgt_labels = targets[b]["labels"]

            if tgt_boxes.numel() == 0:
                # no objects in this image ➜ everything stays background
                indices.append(
                    (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                )
                continue

            # ----- classification cost  ------------------------------------
            out_prob = pred_logits[b].softmax(-1)  # (Q, C+1)
            cost_cls = -out_prob[:, tgt_labels].log()  # (Q, T)

            # ----- bbox & GIoU costs ---------------------------------------
            out_bbox = pred_boxes[b]  # (Q, 4)
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)  # (Q, T)

            # convert to xyxy for IoU
            def cxcywh_to_xyxy(box):
                cx, cy, w, h = box.unbind(-1)
                return torch.stack(
                    [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
                )

            giou = box_iou(cxcywh_to_xyxy(out_bbox), cxcywh_to_xyxy(tgt_boxes))[
                0
            ]  # (Q, T)
            cost_giou = 1.0 - giou

            # ----- final cost matrix ---------------------------------------
            C = (
                self.lambda_cls * cost_cls
                + self.lambda_bbox * cost_bbox
                + self.lambda_giou * cost_giou
            )
            C = C.cpu()

            q_idx, t_idx = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(q_idx, dtype=torch.int64, device=device),
                    torch.as_tensor(t_idx, dtype=torch.int64, device=device),
                )
            )

        return indices


import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    p = F.softmax(inputs, -1)
    ce = F.cross_entropy(inputs, targets, reduction="none")
    pt = p[torch.arange(len(targets), device=inputs.device), targets]

    alpha_factor = torch.full_like(pt, 1 - alpha)
    alpha_factor[targets == inputs.shape[-1] - 1] = alpha

    loss = alpha_factor * (1 - pt) ** gamma * ce
    return loss.mean()


def box_cxcywh_to_xyxy(x):
    # Converts [cx, cy, w, h] -> [x_min, y_min, x_max, y_max]
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_iou(boxes1, boxes2):
    """
    Generalized IoU from the DETR repo.
    Expects boxes1, boxes2 in [x_min, y_min, x_max, y_max] format.
    """
    # Intersection area
    inter_xmin = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_ymin = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_xmax = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_ymax = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter_area

    iou = inter_area / union.clamp(min=1e-6)

    # Compute the enclosed area (smallest box covering both boxes)
    enclosed_xmin = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclosed_ymin = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclosed_xmax = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclosed_ymax = torch.max(boxes1[:, 3], boxes2[:, 3])
    enclosed_w = (enclosed_xmax - enclosed_xmin).clamp(min=0)
    enclosed_h = (enclosed_ymax - enclosed_ymin).clamp(min=0)
    enclosed_area = enclosed_w * enclosed_h

    # Generalized IoU
    giou = iou - (enclosed_area - union) / enclosed_area.clamp(min=1e-6)
    return giou


class DETRLoss(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        """
        Args:
            num_classes: number of *foreground* classes (not counting background).
            matcher:    your Hungarian matcher instance.
            weight_dict: dictionary of loss coefficients, e.g.:
                         {
                           "loss_ce": 1.0,
                           "loss_bbox": 5.0,
                           "loss_giou": 2.0,
                           "loss_cardinlaity": 1.0
                         }
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.register_buffer("ce_weight", torch.tensor([1] * num_classes + [0.1]))

        # DETR typically computes four main losses:
        #   1) classification loss (cross-entropy)
        #   2) bounding box L1 loss
        #   3) GIoU loss
        #   4) cardinality loss (L1 on the # predicted boxes vs. # ground-truth boxes)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: {
               "pred_logits": (B, num_queries, num_classes+1),
               "pred_boxes":  (B, num_queries, 4)
            }
            targets: list of length B, each is a dict with
               {
                 "labels": (num_gt,),
                 "boxes":  (num_gt, 4) in [cx, cy, w, h] normalized coords
               }
        Returns:
            total_loss, loss_dict
              where loss_dict includes the individual losses
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        device = pred_logits.device

        # Hungarian matching to match queries <-> ground truth
        indices = self.matcher(pred_logits, pred_boxes, targets)

        # Flatten predictions for classification
        # shape: (B * num_queries, num_classes+1)
        out_logits = pred_logits.view(-1, self.num_classes + 1)

        # Initialize all queries as "background"
        tgt_labels = torch.full(
            (pred_logits.size(0), pred_logits.size(1)),
            self.num_classes,  # index of background class
            dtype=torch.long,
            device=device,
        )

        # We also keep matched boxes to compute box regression
        # shape: (B * num_queries, 4)
        out_boxes = pred_boxes.view(-1, 4)
        tgt_boxes_for_loss = torch.zeros_like(out_boxes)

        # Fill matched queries with ground truth labels/boxes
        for b_idx, (pred_ids, tgt_ids) in enumerate(indices):
            # matched queries: set ground-truth label instead of background
            tgt_labels[b_idx, pred_ids] = targets[b_idx]["labels"][tgt_ids]

            # matched ground-truth boxes
            matched_gt_boxes = targets[b_idx]["boxes"][tgt_ids]  # (M, 4)
            for i_pred, i_gt in zip(pred_ids, range(len(tgt_ids))):
                global_idx = b_idx * pred_logits.size(1) + i_pred
                tgt_boxes_for_loss[global_idx] = matched_gt_boxes[i_gt]

        # Flatten the labels
        # shape: (B * num_queries,)
        tgt_labels = tgt_labels.view(-1)

        # --------------------------
        # 1) Classification loss
        # --------------------------
        # loss_ce = F.cross_entropy(out_logits, tgt_labels)
        print("out_logits", out_logits)
        print("tgt_labels", tgt_labels)
        print("out_logits.shape", out_logits.shape)
        print("tgt_labels.shape", tgt_labels.shape)

        probs = out_logits.softmax(-1)
        print("avg background prob:", probs[:, -1].mean().item())
        print("max foreground prob:", probs[:, :-1].max(dim=1).values.max().item())
        
        loss_ce = focal_loss(out_logits, tgt_labels, alpha=0.5, gamma=2)

        # --------------------------
        # 2) BBox L1 loss (matched only)
        # --------------------------
        matched_indices = tgt_labels != self.num_classes
        matched_out_boxes = out_boxes[matched_indices]
        matched_tgt_boxes = tgt_boxes_for_loss[matched_indices]
        loss_bbox = F.l1_loss(matched_out_boxes, matched_tgt_boxes, reduction="mean")

        # --------------------------
        # 3) GIoU loss (matched only)
        # --------------------------
        matched_out_xyxy = box_cxcywh_to_xyxy(matched_out_boxes)
        matched_tgt_xyxy = box_cxcywh_to_xyxy(matched_tgt_boxes)
        loss_giou = 1.0 - generalized_iou(matched_out_xyxy, matched_tgt_xyxy)
        loss_giou = loss_giou.mean()

        # --------------------------
        # 4) Cardinality loss
        # --------------------------
        # For each image, count how many queries *are not* predicted as background.
        # 'num_classes' is the background index if your total classes = num_classes+1
        B, num_queries, _ = pred_logits.shape
        pred_class_indices = pred_logits.argmax(dim=2)  # (B, num_queries)
        # predicted # of objects in each image
        pred_counts = (pred_class_indices != self.num_classes).sum(dim=1).float()

        # ground-truth # of boxes in each image
        tgt_counts = torch.as_tensor(
            [len(t["labels"]) for t in targets], dtype=torch.float, device=device
        )

        # L1 loss between predicted count and actual count
        loss_cardinality = F.l1_loss(pred_counts, tgt_counts)

        # Combine into a dictionary
        loss_dict = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_cardinality": loss_cardinality,
        }

        # Weighted sum of all losses
        total_loss = 0.0
        for k, v in loss_dict.items():
            # default weight is 1.0 if not in weight_dict
            weight = self.weight_dict.get(k, 1.0)
            total_loss += weight * v

        return total_loss, loss_dict


train_loader, test_loader = create_coco_dataloaders(batch_size=16)

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
criterion = DETRLoss(
    num_classes=91,
    matcher=HungarianMatcher(),
    weight_dict={
        "loss_ce": 3,
        "loss_bbox": 5,
        "loss_giou": 2.0,
        "loss_cardinality": 0.5,
    },
).to(device)


start_epoch = 0
model_path = "checkpoint.pth"

# Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
scaler = GradScaler()

# Try loading checkpoint
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        exit(1)
else:
    print("Starting training from scratch.")

for epoch in range(start_epoch, 500):
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    total_loss_card = 0.0
    model.train()

    train_loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False)

    for images, targets in train_loop:
        tensors, masks = images.decompose()
        tensors = tensors.to(device)
        masks = masks.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(tensors)
            try:
                loss, loss_dict = criterion(outputs, targets)
            except Exception as e:
                print(outputs)
                print(targets)
                print(e)
                exit(1)

        if not torch.isfinite(loss):
            print("Non-finite loss, skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss values
        total_loss += loss.item()
        total_loss_ce += loss_dict["loss_ce"].item()
        total_loss_bbox += loss_dict["loss_bbox"].item()
        total_loss_giou += loss_dict["loss_giou"].item()
        total_loss_card += loss_dict["loss_cardinality"].item()

        # Update progress bar
        train_loop.set_postfix(
            loss=loss.item(),
            loss_ce=loss_dict["loss_ce"].item(),
            loss_bbox=loss_dict["loss_bbox"].item(),
            loss_giou=loss_dict["loss_giou"].item(),
            loss_cardinality=loss_dict["loss_cardinality"].item(),
        )

    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_loss_ce = total_loss_ce / num_batches
    avg_loss_bbox = total_loss_bbox / num_batches
    avg_loss_giou = total_loss_giou / num_batches
    avg_loss_card = total_loss_card / num_batches

    scheduler.step()

    # Save losses and checkpoint
    with open("losses.txt", "a") as f:
        f.write(
            f"Epoch: {epoch} "
            f"Loss: {avg_loss:.4f} "
            f"CE: {avg_loss_ce:.4f} "
            f"BBox: {avg_loss_bbox:.4f} "
            f"GIoU: {avg_loss_giou:.4f} "
            f"Card: {avg_loss_card:.4f}\n"
        )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        model_path,
    )

    print(
        f"Epoch {epoch+1}, "
        f"Avg Loss: {avg_loss:.4f}, "
        f"CE: {avg_loss_ce:.4f}, "
        f"BBox: {avg_loss_bbox:.4f}, "
        f"GIoU: {avg_loss_giou:.4f}, "
        f"Card: {avg_loss_card:.4f}"
    )
