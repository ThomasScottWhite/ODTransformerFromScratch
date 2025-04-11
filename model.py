import torch.nn as nn
from timm import create_model
import torch
import torchvision.ops as ops

import torch
import torch.nn as nn
import torch
import torch.nn as nn
from transformers import TimesformerConfig, TimesformerModel, ViTModel
from art import FuzzyARTMAP, FuzzyART
from timm import create_model
import torch
import torchvision.ops as ops
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for efficient adaptation in neural networks.
    """

    def __init__(self, model_dim, rank, adapter_size=None):
        """
        Initializes the LoRALayer.

        Parameters:
        - model_dim (int): Dimension of the input features.
        - rank (int): Rank of the low-rank adaptation.
        - adapter_size (int): Size of the adapter layer. Defaults to model_dim.
        """
        super(LoRALayer, self).__init__()
        self.rank = rank
        if adapter_size is None:
            adapter_size = model_dim
        self.down = nn.Linear(model_dim, rank, bias=False)
        self.up = nn.Linear(rank, adapter_size, bias=False)

    def forward(self, x, active=None):
        """
        Forward pass through the LoRALayer.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - active (torch.Tensor): Optional mask for activation.

        Returns:
        - torch.Tensor: Output tensor after applying the LoRA layer.
        """
        if active is not None:
            return torch.where(active.unsqueeze(-1), self.up(self.down(x)), x)
        else:
            return x


class MixtureOfAdapters(nn.Module):
    """
    Mixture of Adapters with LoRA for dynamic feature adaptation.
    """

    def __init__(self, input_dim, output_dim, num_adapters, rank, modality="image"):
        """
        Initializes the MixtureOfAdapters.

        Parameters:
        - input_dim (int): Dimension of the input features.
        - output_dim (int): Dimension of the output features.
        - num_adapters (int): Number of LoRA-based adapters.
        - rank (int): Rank of the low-rank adaptation.
        """
        super(MixtureOfAdapters, self).__init__()
        self.modality = modality
        self.adapters = nn.ModuleList(
            [LoRALayer(input_dim, rank, output_dim) for _ in range(num_adapters)]
        )
        self.weights = nn.Parameter(torch.ones(num_adapters))

    def add_experts(self, num_new_experts, input_dim, output_dim, rank):
        """
        Dynamically adds new LoRA experts to the mixture.

        Parameters:
        - num_new_experts (int): Number of new experts to add.
        - input_dim (int): Input dimension for the new experts.
        - output_dim (int): Output dimension for the new experts.
        - rank (int): Rank of adaptation for the new experts.
        """
        new_adapters = [
            LoRALayer(input_dim, rank, output_dim).to(self.weights.device)
            for _ in range(num_new_experts)
        ]
        self.adapters.extend(new_adapters)
        self.weights = nn.Parameter(
            torch.cat(
                (self.weights, torch.ones(num_new_experts, device=self.weights.device)),
                0,
            )
        )
        print(f"\nAdded {num_new_experts} new experts, total now: {len(self.adapters)}")

    def forward(self, x, gating_scores):
        """
        x: (B, Seq_Len, input_dim)        -> (2, 4165, 768)
        gating_scores: (B, Seq_Len, num_adapters) -> (2, 4165, 2)
        """
        adapter_outputs = torch.stack([adapter(x) for adapter in self.adapters], dim=1)

        gating_scores_expanded = gating_scores.permute(0, 2, 1).unsqueeze(-1)

        weighted_outputs = torch.sum(gating_scores_expanded * adapter_outputs, dim=1)

        return weighted_outputs


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)


class DetectionHead(nn.Module):
    """
    Simple object detection head for bounding boxes and classification.
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.cls_head = nn.Linear(input_dim, num_classes)  # Class predictions
        self.box_head = nn.Linear(input_dim, 4)  # Bounding box predictions
        self.obj_head = nn.Linear(input_dim, 1)  # Object confidence

    def forward(self, x):
        class_logits = self.cls_head(x)  # (B, Seq, num_classes)
        box_preds = self.box_head(x).sigmoid()  # (B, Seq, 4)
        obj_scores = self.obj_head(x)  # (B, Seq, 1) -> Confidence
        return class_logits, box_preds, obj_scores


class MINDObjectDetector(nn.Module):
    """
    MIND model integrating both Video (TimeSformer) and Image (ViT) backbones.
    """

    def __init__(
        self,
        input_size,
        num_heads,
        dynamic_categories,
        rank,
        mode="supervised",
        adaptable_moe=False,
        initial_vigilance=0.75,
        vigilance_increment=0.05,
        modality="image",
    ):
        super().__init__()
        self.mode = mode
        self.adaptable_moe = adaptable_moe
        self.rank = rank
        self.modality = modality
        self.num_categories = dynamic_categories

        self.feature_dim = 768  # TimeSformer output size

        if mode == "supervised":
            self.art_module = FuzzyARTMAP(
                input_dim=self.feature_dim,
                dynamic_categories=dynamic_categories,
                initial_vigilance=initial_vigilance,
                vigilance_increment=vigilance_increment,
            )
        else:
            self.art_module = FuzzyART(
                input_dim=self.feature_dim,
                dynamic_categories=dynamic_categories,
                initial_vigilance=initial_vigilance,
                vigilance_increment=vigilance_increment,
            )

        if self.adaptable_moe:
            num_initial_experts = 2
            self.gating_network = GatingNetwork(self.feature_dim, num_initial_experts)
            self.current_expert_count = num_initial_experts
            self.moe_adapters = MixtureOfAdapters(
                self.feature_dim,
                self.feature_dim,
                num_initial_experts,
                rank,
                modality=self.modality,
            )
            self.expand_experts()
        else:
            max_categories = dynamic_categories
            self.gating_network = GatingNetwork(self.feature_dim, max_categories)
            self.moe_adapters = MixtureOfAdapters(
                self.feature_dim,
                self.feature_dim,
                max_categories,
                rank,
                modality=self.modality,
            )
            self.expand_experts()

        self.backbone = create_model(
            "swin_base_patch4_window7_224", pretrained=True, features_only=True
        )
        self.projection_C2 = nn.Linear(128, 768)
        self.projection_C3 = nn.Linear(256, 768)
        self.projection_C4 = nn.Linear(512, 768)
        self.projection_C5 = nn.Linear(1024, 768)

        # Freeze pretrained backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.detection_head = DetectionHead(
            input_dim=768, num_classes=self.num_categories
        )

    # def forward(self, x):
    #     C2, C3, C4, C5 = self.backbone(x)

    #     C2_flat = self.flatten_features(C2)
    #     C3_flat = self.flatten_features(C3)
    #     C4_flat = self.flatten_features(C4)
    #     C5_flat = self.flatten_features(C5)

    #     C2_proj = self.projection_C2(C2_flat)
    #     C3_proj = self.projection_C3(C3_flat)
    #     C4_proj = self.projection_C4(C4_flat)
    #     C5_proj = self.projection_C5(C5_flat)

    #     moe_input = torch.cat([C2_proj, C3_proj, C4_proj, C5_proj], dim=1)

    #     if self.adaptable_moe:
    #         self.expand_experts()

    #     gating_scores = self.gating_network(moe_input)
    #     moe_output = self.moe_adapters(moe_input, gating_scores)

    #     class_logits, box_preds, obj_scores = self.detection_head(moe_output)

    #     img_h, img_w = 224, 224
    #     scale = torch.tensor([img_w, img_h, img_w, img_h], device=box_preds.device)
    #     box_preds = box_preds * scale

    #     return class_logits, box_preds, obj_scores

    # Idk maybe this is better
    def forward(self, x):
        _, _, C4, _ = self.backbone(x)

        C4_flat = self.flatten_features(C4)
        C4_proj = self.projection_C4(C4_flat)

        moe_input = C4_proj

        if self.adaptable_moe:
            self.expand_experts()

        gating_scores = self.gating_network(moe_input)
        moe_output = self.moe_adapters(moe_input, gating_scores)

        class_logits, box_preds, obj_scores = self.detection_head(moe_output)

        # Convert boxes from absolute xyxy to normalized cxcywh
        def xyxy_to_cxcywh(boxes):
            x1, y1, x2, y2 = boxes.unbind(-1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            return torch.stack((cx, cy, w, h), dim=-1)

        # Assuming input images are always 224x224
        img_h, img_w = 640, 224
        scale = torch.tensor([img_w, img_h, img_w, img_h], device=box_preds.device)

        box_preds = xyxy_to_cxcywh(box_preds)  # Convert format
        box_preds = box_preds / scale          # Normalize to [0, 1]

        return {
            "pred_logits": class_logits,
            "pred_boxes": box_preds,
            "pred_obj": obj_scores,
        }

    def apply_nms_per_image(self, class_logits, boxes, obj_scores, conf_mask):
        iou_threshold = 0.5
        batch_size = class_logits.shape[0]

        final_boxes, final_scores, final_labels = [], [], []

        for batch_idx in range(batch_size):
            valid_boxes = boxes[batch_idx][conf_mask[batch_idx]]
            valid_scores = obj_scores[batch_idx][conf_mask[batch_idx]]
            valid_class_logits = class_logits[batch_idx][conf_mask[batch_idx]]

            if valid_boxes.numel() == 0:
                final_boxes.append([])
                final_scores.append([])
                final_labels.append([])
                continue

            class_probs = valid_class_logits.softmax(dim=-1)
            class_scores, class_labels = class_probs.max(dim=-1)

            final_scores_for_nms = valid_scores * class_scores
            keep_indices = ops.nms(valid_boxes, final_scores_for_nms, iou_threshold)

            final_boxes.append(valid_boxes[keep_indices])
            final_scores.append(final_scores_for_nms[keep_indices])
            final_labels.append(class_labels[keep_indices])

        return final_boxes, final_scores, final_labels

    def expand_experts(self):
        """
        Expands the number of experts if we have discovered more categories than existing experts.
        """
        if not self.adaptable_moe:
            return
        current_categories = self.art_module.get_current_categories()
        if current_categories > self.current_expert_count:
            new_experts_needed = current_categories - self.current_expert_count + 1
            self.moe_adapters.add_experts(
                num_new_experts=new_experts_needed,
                input_dim=self.feature_dim,
                output_dim=self.feature_dim,
                rank=self.rank,
            )
            self.gating_network.fc = nn.Linear(
                self.feature_dim, current_categories + 1
            ).to(self.gating_network.fc.weight.device)
            self.current_expert_count = current_categories + 1

    def flatten_features(self, features):
        B, H, W, D = features.shape
        return features.view(B, H * W, D)

    def state_dict(self, *args, **kwargs):
        """
        Returns a dictionary containing the state of the model,
        including ART categories.
        """
        state = super().state_dict(*args, **kwargs)
        state["art_module.categories"] = self.art_module.categories
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Loads the model state, including ART categories if present.
        """
        if "art_module.categories" in state_dict:
            self.art_module.categories = state_dict["art_module.categories"]
            del state_dict["art_module.categories"]
        super().load_state_dict(state_dict, *args, **kwargs)
