from MIND import MINDObjectDetector
from datasets.coco import create_coco_dataloaders
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MINDObjectDetector(
    input_size=224,
    num_heads=12,
    dynamic_categories=80,
    rank=32,
    mode="supervised",
    adaptable_moe=True,
    initial_vigilance=0.75,
    vigilance_increment=0.05,
    modality="object_detection",
).to(device)

train_loader, test_loader = create_coco_dataloaders(
    root_dir="/home/thomas/acil/ODMind/MIND/data/coco2017",
    batch_size=2,
    num_workers=1,
    trim_fraction=1,
)

import torch.nn.functional as F
from torchvision.ops import box_iou


from torchmetrics.detection.mean_ap import MeanAveragePrecision


def evaluate(model, dataloader, iou_threshold=0.5):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)

    with torch.no_grad():
        for img, boxes, labels in tqdm(dataloader, desc="Evaluating"):
            img = img.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            class_logits, box_preds, obj_scores = model(img)

            for i in range(len(img)):
                pred_scores = obj_scores[i].squeeze(-1).cpu()
                pred_labels = class_logits[i].argmax(dim=1).cpu()
                pred_boxes = box_preds[i].cpu()

                gt_boxes = boxes[i].cpu()
                gt_labels = labels[i].cpu()

                # Required format for torchmetrics
                preds = [
                    {
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels,
                    }
                ]

                targets = [
                    {
                        "boxes": gt_boxes,
                        "labels": gt_labels,
                    }
                ]

                metric.update(preds, targets)

    results = metric.compute()
    model.train()
    return results


import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Computes Focal Loss for classification.

    Args:
        logits (Tensor): shape (N, C), raw output from the model (before softmax)
        targets (Tensor): shape (N,), integer class labels
        alpha (float): balancing factor
        gamma (float): focusing parameter
        reduction (str): 'mean', 'sum', or 'none'

    Returns:
        Tensor: scalar loss
    """
    ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (N,)
    pt = torch.exp(-ce_loss).clamp(
        min=1e-4, max=1.0
    )  # Prevents nans when probability 0
    focal_term = alpha * (1 - pt) ** gamma

    loss = focal_term * ce_loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def compute_loss(class_logits, box_preds, obj_scores, gt_boxes, gt_labels):
    batch_size = class_logits.size(0)
    total_cls_loss, total_box_loss, total_obj_loss = 0.0, 0.0, 0.0

    for i in range(batch_size):
        pred_logits = class_logits[i]
        pred_boxes = box_preds[i]
        pred_scores = obj_scores[i].squeeze(-1)
        gt_b = gt_boxes[i].to(device)
        gt_l = gt_labels[i].to(device)
        gt_l = gt_l.squeeze(-1) if gt_l.dim() > 1 else gt_l

        if gt_b.numel() == 0 or gt_l.numel() == 0:
            cls_loss = torch.tensor(0.0, device=pred_logits.device)
            box_loss = torch.tensor(0.0, device=pred_logits.device)
            obj_loss = F.binary_cross_entropy_with_logits(
                pred_scores, torch.zeros_like(pred_scores), reduction="none"
            )
            pt = torch.exp(-obj_loss).clamp(min=1e-4, max=1.0)
            focal_term = 0.25 * (1 - pt) ** 2.0
            obj_loss = (focal_term * obj_loss).mean()
        else:
            ious = box_iou(pred_boxes, gt_b)
            max_ious, matched_gt = ious.max(dim=1)
            matched_labels = gt_l[matched_gt]
            matched_mask = max_ious > 0.5

            obj_target = matched_mask.float()

            if matched_mask.sum() > 0:
                cls_loss = focal_loss(
                    pred_logits[matched_mask],
                    matched_labels[matched_mask],
                    alpha=0.25,
                    gamma=2.0,
                )
                box_loss = F.l1_loss(
                    pred_boxes[matched_mask], gt_b[matched_gt[matched_mask]]
                )
            else:
                cls_loss = torch.tensor(0.0, device=pred_logits.device)
                topk = min(5, pred_boxes.shape[0])
                topk_iou, topk_idx = max_ious.topk(topk)
                fallback_preds = pred_boxes[topk_idx]
                fallback_gt = gt_b[matched_gt[topk_idx]]
                box_loss = F.l1_loss(fallback_preds, fallback_gt)

            obj_loss = F.binary_cross_entropy_with_logits(
                pred_scores, obj_target, reduction="none"
            )
            pt = torch.exp(-obj_loss)
            focal_term = 0.25 * (1 - pt) ** 2.0
            obj_loss = (focal_term * obj_loss).mean()

        total_cls_loss += cls_loss
        total_box_loss += box_loss
        total_obj_loss += obj_loss

    loss = total_cls_loss + 5.0 * total_box_loss + 1.0 * total_obj_loss
    return loss, total_cls_loss, total_box_loss, total_obj_loss


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
from tqdm import tqdm

for epoch in range(100):
    train_loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False)
    total_loss = 0
    total_cls_loss = 0
    total_box_loss = 0
    total_obj_loss = 0

    for img, boxes, labels in train_loop:
        img = img.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        optimizer.zero_grad()
        class_logits, box_preds, obj_scores = model(img)

        loss, cls_loss, box_loss, obj_loss = compute_loss(
            class_logits, box_preds, obj_scores, boxes, labels
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_cls_loss += cls_loss.detach()
        total_box_loss += box_loss.detach()
        total_obj_loss += obj_loss.detach()

        train_loop.set_postfix(
            loss=loss.item(),
            cls_loss=cls_loss,
            box_loss=box_loss,
            obj_loss=obj_loss,
        )

    loss = total_loss / len(train_loader)
    cls_loss = total_cls_loss / len(train_loader)
    box_loss = total_box_loss / len(train_loader)
    obj_loss = total_obj_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {loss:.4f}")
    print(f"Epoch {epoch+1} Classification Loss: {cls_loss:.4f}")
    print(f"Epoch {epoch+1} Box Loss: {box_loss:.4f}")
    print(f"Epoch {epoch+1} Objectness Loss: {obj_loss:.4f}")

    # Evaluate the model
    results = evaluate(model, test_loader)
    print(f"MAP 50 Results: {results}")

    # Save the model checkpoint
    torch.save(model.state_dict(), f"model.pth")
