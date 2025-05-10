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

import math
import torch


def build_2d_sincos_position_embedding(H, W, dim, temperature=10000.0):
    """
    Build 2D sine-cosine positional embeddings like in DETR.

    Args:
        H, W: height and width of feature map
        dim: total embedding dimension (must be divisible by 4)
    Returns:
        Tensor of shape [H * W, dim]
    """
    if dim % 4 != 0:
        raise ValueError("dim must be divisible by 4")

    # Each position gets dim // 2 (split across X and Y)
    dim_each = dim // 2
    dim_x = dim_y = dim_each // 2  # Half for x, half for y

    y_embed = torch.linspace(0, 1, steps=H)
    x_embed = torch.linspace(0, 1, steps=W)
    grid_y, grid_x = torch.meshgrid(y_embed, x_embed, indexing="ij")

    # [H, W] → [H, W, dim/4]
    omega_x = 1.0 / (temperature ** (torch.arange(dim_x, dtype=torch.float32) / dim_x))
    omega_y = 1.0 / (temperature ** (torch.arange(dim_y, dtype=torch.float32) / dim_y))

    pos_x = grid_x[..., None] * omega_x  # [H, W, dim/4]
    pos_y = grid_y[..., None] * omega_y  # [H, W, dim/4]

    pos_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=-1)  # [H, W, dim/2]
    pos_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=-1)  # [H, W, dim/2]

    pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, dim]
    pos = pos.view(H * W, dim)  # [H*W, dim]

    return pos


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
        return torch.softmax(self.fc(x), dim=-1)


import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    A more expressive object detection head with MLPs for classification and bounding box regression.
    """

    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()

        # Classification head: 2-layer MLP + LayerNorm
        self.cls_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes + 1),  # +1 for background
        )

        # Box regression head: 3-layer MLP + LayerNorm
        self.box_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: [B, Seq, input_dim]
        Returns:
            class_logits: [B, Seq, num_classes + 1]
            box_preds: [B, Seq, 4]
        """
        class_logits = self.cls_head(x)
        box_preds = self.box_head(x)
        return class_logits, box_preds


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

        dims = [f["num_chs"] for f in self.backbone.feature_info]
        print("Feature map dimensions:", dims)
        self.projection_C2 = nn.Linear(dims[0], 768)
        self.projection_C3 = nn.Linear(dims[1], 768)
        self.projection_C4 = nn.Linear(dims[2], 768)
        self.projection_C5 = nn.Linear(dims[3], 768)

        self.level_embed = nn.Parameter(torch.zeros(4, 1, 768))

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.detection_head = DetectionHead(
            input_dim=768, num_classes=self.num_categories
        )

    def forward(self, x):
        """
        Multi‑scale forward pass
        ------------------------
        1. Backbone → {C2,C3,C4,C5}
        2. Flatten + project each scale to a common embed_dim
        3. Add 2‑D sin‑cos positional encodings  +  a learnable level‑embedding
        4. Concatenate all tokens     →  [B, ΣHW, embed_dim]
        5. MoE routing + detection head
        """
        B = x.size(0)

        C2, C3, C4, C5 = self.backbone(x)

        total_tokens = 100
        num_levels = 4
        tokens_per_level = total_tokens // num_levels  # e.g. 25 per level

        # Adjust to your feature maps
        feature_maps = [C2, C3, C4, C5]
        projections = [
            self.projection_C2,
            self.projection_C3,
            self.projection_C4,
            self.projection_C5,
        ]

        tokens_per_level_list = []

        for lvl, (feat_map, proj, lvl_embed) in enumerate(zip(feature_maps, projections, self.level_embed)):
            # Convert [B, H, W, C] → [B, C, H, W] if needed
            if feat_map.shape[-1] == proj.in_features:
                feat_map = feat_map.permute(0, 3, 1, 2)  # NHWC → NCHW

            B, C, H, W = feat_map.shape

            # Downsample to fixed grid (e.g., 5x5 = 25 tokens per level)
            target_h = target_w = int(tokens_per_level ** 0.5)
            pooled = F.adaptive_avg_pool2d(feat_map, output_size=(target_h, target_w))  # [B, C, h, w]

            # Flatten spatial and permute to [B, HW, C]
            pooled = pooled.flatten(2).permute(0, 2, 1)  # [B, tokens, C]

            # Project to embed dim
            tokens = proj(pooled)  # [B, tokens, D]

            # Positional encoding for each grid
            pos = build_2d_sincos_position_embedding(target_h, target_w, tokens.size(-1)).to(tokens.device)
            pos = pos.unsqueeze(0).expand(B, -1, -1)

            tokens = tokens + pos + lvl_embed  # Add positional + level encoding

            tokens_per_level_list.append(tokens)

        # Final compressed multiscale token sequence: [B, ~100, D]
        moe_input = torch.cat(tokens_per_level_list, dim=1)

        # 5️⃣ Routing through Mixture-of-Experts (with optional dynamic expert growth)
        if self.adaptable_moe:
            self.expand_experts()

        gating_scores = self.gating_network(moe_input)  # [B, ΣHW, E]
        moe_output = self.moe_adapters(moe_input, gating_scores)  # [B, ΣHW, D]

        # Final detection head
        class_logits, box_preds = self.detection_head(moe_output)
        return {
            "pred_logits": class_logits,  # [B, ΣHW, num_classes + 1]
            "pred_boxes": box_preds,  # [B, ΣHW, 4]
        }

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

            # Sync gating network to new adapter count
            new_total = len(self.moe_adapters.adapters)
            self.gating_network.fc = nn.Linear(self.feature_dim, new_total).to(
                self.gating_network.fc.weight.device
            )

            self.current_expert_count = new_total

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
