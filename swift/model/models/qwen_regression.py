# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Qwen3-VL Multi-Task Regression Model with Early Fusion

This module provides a custom model for multi-task regression based on Qwen3-VL-Embedding,
with support for:
- Multi-scale point cloud feature encoding (512-dim input → 3 scale outputs)
- DeepStack-style early fusion at LLM layers 8, 16, 24
- Point cloud embedding prepended to sequence as global context
- Cross-Attention fusion for final representation
- Multiple parallel regression heads
- Weighted MSE loss for task balancing

Architecture:
```
Point Cloud (512-dim) → Multi-scale Encoder → [pc_embed_0, pc_embed_1, pc_embed_2]
                                                    ↓
Sequence: [pc_embed_0] + [text tokens + image tokens]
                                                    ↓
LLM Forward with DeepStack injection at layers 8, 16, 24
                                                    ↓
First Token (point cloud context) → Regression Heads
```
"""
import math
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

from swift.model.model_arch import ModelArch
from swift.model.model_meta import Model, ModelGroup, ModelMeta
from swift.model.models.qwen import Qwen3VLLoader
from swift.model.register import register_model
from swift.template import TemplateType
from swift.utils import get_logger

logger = get_logger()

# DeepStack injection layers (same as visual features)
DEEPSTACK_LAYERS = [8, 16, 24]


class PointCloudMultiScaleEncoder(nn.Module):
    """
    Multi-Scale Point Cloud Encoder for DeepStack-style early fusion.

    Outputs embeddings at 3 different scales, each to be injected at
    a different LLM layer (8, 16, 24) similar to how Qwen3-VL handles
    multi-scale visual features.

    Args:
        input_dim: Dimension of input point cloud features (default: 512)
        hidden_size: Target hidden dimension to match LLM (default: 1536)
        num_scales: Number of scale outputs (default: 3 for layers 8, 16, 24)
        intermediate_dim: Hidden dimension in scale layers (default: 1024)
        dropout: Dropout probability (default: 0.1)
        dtype: Data type for parameters (default: torch.bfloat16)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_size: int = 1536,
        num_scales: int = 3,
        intermediate_dim: int = 1024,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        self.num_scales = num_scales
        self.hidden_size = hidden_size

        # Shared initial projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size, dtype=dtype),
            nn.LayerNorm(hidden_size, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Scale-specific layers (each produces an embedding for a different LLM layer)
        self.scale_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_dim, dtype=dtype),
                nn.LayerNorm(intermediate_dim, dtype=dtype),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_size, dtype=dtype),
                nn.LayerNorm(hidden_size, dtype=dtype),
            ) for _ in range(num_scales)
        ])

    def forward(self, point_cloud_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for multi-scale point cloud encoding.

        Args:
            point_cloud_features: Input tensor of shape [batch_size, input_dim]

        Returns:
            List of num_scales tensors, each of shape [batch_size, hidden_size]
        """
        # Shared projection
        x = self.input_projection(point_cloud_features)  # [batch, hidden_size]

        # Generate scale-specific outputs
        outputs = []
        for scale_layer in self.scale_layers:
            scale_out = scale_layer(x)  # [batch, hidden_size]
            outputs.append(scale_out)

        return outputs  # List of [batch, hidden_size] tensors


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module for final representation.

    Fuses point cloud features and image features using cross-attention.
    Used after the LLM forward pass for final regression.

    Args:
        hidden_size: Dimension of hidden representations (default: 1536)
        num_attention_heads: Number of attention heads (default: 12)
        attention_probs_dropout_prob: Dropout for attention probabilities (default: 0.1)
        hidden_dropout_prob: Dropout for output (default: 0.1)
        dtype: Data type for parameters (default: torch.bfloat16)
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        assert hidden_size == self.all_head_size, \
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"

        self.query = nn.Linear(hidden_size, self.all_head_size, dtype=dtype)
        self.key = nn.Linear(hidden_size, self.all_head_size, dtype=dtype)
        self.value = nn.Linear(hidden_size, self.all_head_size, dtype=dtype)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.output = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, dtype=dtype)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        point_cloud_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if point_cloud_embeds.dim() == 2:
            point_cloud_embeds = point_cloud_embeds.unsqueeze(1)
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(1)

        query_layer = self.transpose_for_scores(self.query(point_cloud_embeds))
        key_layer = self.transpose_for_scores(self.key(image_embeds))
        value_layer = self.transpose_for_scores(self.value(image_embeds))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        output = self.output(context_layer)
        output = self.output_dropout(output)
        fused = self.layer_norm(point_cloud_embeds + output)

        return fused.squeeze(1)


class MultiTaskRegressionHead(nn.Module):
    """
    Multi-Task Regression Head.

    Each regression task has an independent MLP head mapping from hidden_size
    to a single scalar output.

    Args:
        hidden_size: Input hidden dimension (default: 1536)
        num_tasks: Number of regression tasks (default: 5)
        task_hidden_dims: List of hidden dimensions for each task head (optional)
        dropout: Dropout probability (default: 0.1)
        dtype: Data type for parameters (default: torch.bfloat16)
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_tasks: int = 5,
        task_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        self.num_tasks = num_tasks

        if task_hidden_dims is None:
            task_hidden_dims = [hidden_size // 2] * num_tasks

        assert len(task_hidden_dims) == num_tasks, \
            f"task_hidden_dims length ({len(task_hidden_dims)}) must match num_tasks ({num_tasks})"

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, task_hidden_dims[i], dtype=dtype),
                nn.LayerNorm(task_hidden_dims[i], dtype=dtype),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(task_hidden_dims[i], 1, dtype=dtype)
            ) for i in range(num_tasks)
        ])

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {f'task_{i}': head(embeddings) for i, head in enumerate(self.heads)}


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for Multi-Task Learning.

    Args:
        num_tasks: Number of regression tasks (default: 5)
        task_weights: List of weights for each task (default: uniform weights)
        reduction: Reduction method ('mean' or 'sum', default: 'mean')
    """

    def __init__(
        self,
        num_tasks: int = 5,
        task_weights: Optional[List[float]] = None,
        reduction: str = 'mean'
    ):
        super().__init__()

        if task_weights is None:
            task_weights = [1.0] * num_tasks

        assert len(task_weights) == num_tasks

        self.register_buffer('task_weights', torch.tensor(task_weights, dtype=torch.float32))
        self.reduction = reduction
        self.num_tasks = num_tasks

    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0

        for i in range(self.num_tasks):
            key = f'task_{i}'
            pred = predictions[key]
            target = targets[:, i:i+1]

            mse = F.mse_loss(pred, target, reduction='none')
            if self.reduction == 'mean':
                mse = mse.mean()
            elif self.reduction == 'sum':
                mse = mse.sum()

            total_loss = total_loss + self.task_weights[i] * mse

        return total_loss


class Qwen3VLRegressionOutput(ModelOutput):
    """Output class for Qwen3-VL Multi-Task Regression model."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[Dict[str, torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    fused_embeddings: Optional[torch.Tensor] = None


def _patch_qwen3vl_model_for_pointcloud(base_model, point_cloud_encoder):
    """
    Patch Qwen3VLModel to prepend point cloud embeddings to the sequence
    and pass multi-scale embeddings for DeepStack injection.
    """
    model = base_model.model if hasattr(base_model, 'model') else base_model
    original_forward = model.forward

    def new_forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        point_cloud_features=None,
        point_cloud_embeds=None,  # Pre-computed multi-scale embeddings
        attention_mask=None,
        inputs_embeds=None,  # Accept inputs_embeds for compatibility
        **kwargs
    ):
        # Debug logging
        logger.info(f"[DEBUG] Qwen3VLModel.new_forward: input_ids={'None' if input_ids is None else f'Tensor{input_ids.shape}'}, "
                    f"inputs_embeds={'None' if inputs_embeds is None else f'Tensor{inputs_embeds.shape}'}, "
                    f"kwargs keys: {list(kwargs.keys())}")

        # 1. Process point cloud if provided
        if point_cloud_features is not None and point_cloud_embeds is None:
            point_cloud_embeds = point_cloud_encoder(point_cloud_features)

        # 2. Get text embeddings (prefer inputs_embeds if provided, otherwise from input_ids)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify either input_ids or inputs_embeds")
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 3. Process images (existing Qwen3-VL logic)
        if pixel_values is not None:
            # Get image features
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)

            # Scatter image embeddings into text embeddings
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            deepstack_image_embeds = None

        # 4. Prepend point cloud embedding to sequence (early fusion)
        if point_cloud_embeds is not None:
            # Use first scale embedding for initial sequence
            pc_embed = point_cloud_embeds[0].unsqueeze(1)  # [batch, 1, hidden]
            inputs_embeds = torch.cat([pc_embed, inputs_embeds], dim=1)

            # Adjust attention mask
            if attention_mask is not None:
                pc_mask = torch.ones(
                    attention_mask.shape[0], 1,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([pc_mask, attention_mask], dim=1)

        # 5. Call language model with point cloud embeddings for DeepStack
        # Remove conflicting keys from kwargs
        kwargs.pop('input_ids', None)
        kwargs.pop('inputs_embeds', None)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=kwargs.pop('position_ids', None),
            past_key_values=kwargs.pop('past_key_values', None),
            use_cache=kwargs.pop('use_cache', None),
            output_hidden_states=kwargs.pop('output_hidden_states', True),
            output_attentions=kwargs.pop('output_attentions', None),
            return_dict=kwargs.pop('return_dict', True),
            # Pass for DeepStack injection
            deepstack_visual_embeds=deepstack_image_embeds,
            visual_pos_masks=None,  # Will be computed if needed
            point_cloud_embeds=point_cloud_embeds,  # NEW: for point cloud DeepStack
            **kwargs  # Pass remaining kwargs
        )

        return outputs

    model.forward = MethodType(new_forward, model)


def _patch_language_model_for_pointcloud_deepstack(language_model):
    """
    Patch Qwen3VLTextModel to inject point cloud features at DeepStack layers.

    This follows the same pattern as visual DeepStack but injects point cloud
    features into the first token (point cloud context token) at layers 8, 16, 24.
    """
    original_forward = language_model.forward

    def new_forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
        point_cloud_embeds=None,  # NEW
        input_ids=None,  # Accept input_ids and convert to inputs_embeds
        cache_position=None,  # Accept for compatibility
        **kwargs
    ):
        # Handle input_ids -> inputs_embeds conversion if needed
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            input_ids = None  # Clear to avoid XOR error

        # Store point cloud for layer-wise injection
        self._point_cloud_embeds = point_cloud_embeds
        self._pc_injection_layer = 0

        # Remove input_ids from kwargs if present to avoid conflicts
        kwargs.pop('input_ids', None)

        return original_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs  # Pass remaining kwargs
        )

    language_model.forward = MethodType(new_forward, language_model)

    # Patch the _deepstack_process method to also inject point cloud
    if hasattr(language_model, '_deepstack_process'):
        original_deepstack = language_model._deepstack_process

        def new_deepstack(self, hidden_states, visual_pos_masks, visual_embeds):
            # Original visual deepstack injection
            hidden_states = original_deepstack(hidden_states, visual_pos_masks, visual_embeds)

            # Inject point cloud at corresponding layer
            if self._point_cloud_embeds is not None:
                pc_idx = self._pc_injection_layer
                if pc_idx < len(self._point_cloud_embeds):
                    pc_embed = self._point_cloud_embeds[pc_idx]
                    # Add to first token (point cloud context token)
                    # hidden_states: [seq_len, batch, hidden] or [batch, seq_len, hidden]
                    if hidden_states.dim() == 3:
                        if hidden_states.shape[0] < hidden_states.shape[1]:
                            # [batch, seq, hidden]
                            hidden_states[:, 0, :] = hidden_states[:, 0, :] + pc_embed
                        else:
                            # [seq, batch, hidden]
                            hidden_states[0, :, :] = hidden_states[0, :, :] + pc_embed

                    self._pc_injection_layer += 1

            return hidden_states

        language_model._deepstack_process = MethodType(new_deepstack, language_model)


class Qwen3VLForMultiTaskRegression(nn.Module):
    """
    Qwen3-VL Model for Multi-Task Regression with Early Fusion.

    This model extends Qwen3-VL-Embedding with:
    1. Multi-scale point cloud encoding (3 scales for layers 8, 16, 24)
    2. Early fusion: point cloud prepended to sequence as global context
    3. DeepStack injection: point cloud features added at layers 8, 16, 24
    4. Cross-Attention fusion for final representation (optional)
    5. Multiple parallel regression heads
    6. Weighted MSE loss for task balancing

    This is a wrapper model around Qwen3VLForConditionalGeneration that adds
    point cloud processing and regression heads. It delegates most operations
    to the base_model while providing the necessary interface for training.

    Args:
        base_model: The base Qwen3-VL model
        point_cloud_dim: Dimension of input point cloud features (default: 512)
        hidden_size: Hidden dimension (default: 1536 for Qwen3-VL-2B)
        num_regression_tasks: Number of regression tasks (default: 5)
        task_weights: Weights for each task in loss computation (optional)
        use_cross_attention: Whether to use cross-attention for final fusion (default: True)
        dtype: Data type for new parameters (default: torch.bfloat16)
    """

    # Class-level attributes expected by some trainers/frameworks
    supports_gradient_checkpointing = True
    _no_split_modules = None
    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = None

    def __init__(
        self,
        base_model,
        point_cloud_dim: int = 512,
        hidden_size: int = 1536,
        num_regression_tasks: int = 5,
        task_weights: Optional[List[float]] = None,
        use_cross_attention: bool = True,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        # Store the base model - this is the actual Qwen3VL model
        self.base_model = base_model

        # Initialize instance attributes expected by trainers
        self._warnings_issued = {}
        self._require_grads_hook = None
        self._model_dir = getattr(base_model, 'model_dir', None)
        self._name_or_path = base_model.name_or_path
        self._generation_config = base_model.generation_config

        self.num_regression_tasks = num_regression_tasks
        self.hidden_size = hidden_size
        self.use_cross_attention = use_cross_attention

        # Multi-scale point cloud encoder (outputs 3 embeddings for DeepStack)
        self.point_cloud_encoder = PointCloudMultiScaleEncoder(
            input_dim=point_cloud_dim,
            hidden_size=hidden_size,
            num_scales=len(DEEPSTACK_LAYERS),  # 3 scales for layers 8, 16, 24
            dtype=dtype
        )

        # Apply patches for early fusion
        _patch_qwen3vl_model_for_pointcloud(base_model, self.point_cloud_encoder)
        _patch_language_model_for_pointcloud_deepstack(base_model.model.language_model)

        # Cross-Attention fusion for final representation (optional)
        if use_cross_attention:
            # Use head_size=128 to compute num_attention_heads dynamically
            # This ensures divisibility regardless of model hidden_size
            attention_head_size = 128
            num_attention_heads = hidden_size // attention_head_size
            assert hidden_size % attention_head_size == 0, \
                f"hidden_size ({hidden_size}) must be divisible by attention_head_size ({attention_head_size})"

            self.cross_attention = CrossAttentionFusion(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dtype=dtype
            )
        else:
            self.cross_attention = None

        # Multi-task regression heads
        self.regression_head = MultiTaskRegressionHead(
            hidden_size=hidden_size,
            num_tasks=num_regression_tasks,
            dtype=dtype
        )

        # Loss function
        self.loss_fn = WeightedMSELoss(
            num_tasks=num_regression_tasks,
            task_weights=task_weights
        )

        logger.info(
            f"Initialized Qwen3-VL Multi-Task Regression with early fusion. "
            f"Point cloud will be injected at LLM layers {DEEPSTACK_LAYERS}"
        )

    # =========================================================================
    # Properties with getters and setters
    # =========================================================================

    @property
    def config(self):
        """Config from base model."""
        return self.base_model.config

    @property
    def device(self):
        """Device from base model."""
        return self.base_model.device

    @property
    def dtype(self):
        """Dtype from base model."""
        return self.base_model.dtype

    @property
    def model(self):
        """Return base_model so that model.language_model resolves correctly."""
        return self.base_model

    @property
    def language_model(self):
        """Language model from base model."""
        return self.base_model.language_model

    @property
    def visual(self):
        """Visual encoder from base model."""
        return self.base_model.visual

    @property
    def name_or_path(self):
        """Model name or path."""
        return self._name_or_path

    @name_or_path.setter
    def name_or_path(self, value):
        """Set model name or path."""
        self._name_or_path = value
        if hasattr(self.base_model, 'name_or_path'):
            self.base_model.name_or_path = value

    @property
    def generation_config(self):
        """Generation config."""
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value):
        """Set generation config."""
        self._generation_config = value
        if hasattr(self.base_model, 'generation_config'):
            self.base_model.generation_config = value

    @property
    def model_dir(self):
        """Model directory."""
        return self._model_dir

    @model_dir.setter
    def model_dir(self, value):
        """Set model directory."""
        self._model_dir = value
        if hasattr(self.base_model, 'model_dir'):
            self.base_model.model_dir = value

    @property
    def warnings_issued(self):
        """Warnings issued dict."""
        return self._warnings_issued

    @warnings_issued.setter
    def warnings_issued(self, value):
        """Set warnings issued."""
        self._warnings_issued = value

    # =========================================================================
    # Methods delegated to base_model
    # =========================================================================

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        return None

    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Enable gradient checkpointing in base model."""
        return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        """Disable gradient checkpointing in base model."""
        return self.base_model.gradient_checkpointing_disable(*args, **kwargs)

    def enable_input_require_grads(self):
        """Enable input require grads for gradient checkpointing compatibility."""
        return self.base_model.enable_input_require_grads()

    def get_image_features(self, *args, **kwargs):
        """Get image features from base model."""
        return self.base_model.get_image_features(*args, **kwargs)

    def get_video_features(self, *args, **kwargs):
        """Get video features from base model."""
        return self.base_model.get_video_features(*args, **kwargs)

    def get_rope_index(self, *args, **kwargs):
        """Get rope index for position ids computation.

        get_rope_index is defined in Qwen3VLModel, not Qwen3VLForConditionalGeneration.
        So we need to access it via base_model.model.get_rope_index.
        """
        return self.base_model.model.get_rope_index(*args, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        point_cloud_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, Qwen3VLRegressionOutput]:
        """
        Forward pass for multi-task regression with early fusion.

        Args:
            input_ids: Input token IDs
            pixel_values: Image pixel values
            image_grid_thw: Image grid dimensions
            point_cloud_features: Point cloud features [batch, 512]
            attention_mask: Attention mask
            labels: Regression targets [batch, num_tasks]
            **kwargs: Additional arguments passed to base model

        Returns:
            Qwen3VLRegressionOutput containing loss, predictions, and hidden states
        """
        # 1. Encode point cloud to multi-scale embeddings
        point_cloud_embeds = None
        if point_cloud_features is not None:
            point_cloud_embeds = self.point_cloud_encoder(point_cloud_features)

        # 2. Forward through base model (point cloud is injected early)
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            point_cloud_features=point_cloud_features,
            point_cloud_embeds=point_cloud_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # 3. Get final representations
        last_hidden_state = outputs.last_hidden_state

        # First token is the point cloud context token
        pc_context_embed = last_hidden_state[:, 0, :]

        # Get image/text representation (last valid token)
        if attention_mask is not None:
            # Account for prepended point cloud token
            adjusted_mask = attention_mask[:, 1:] if attention_mask.shape[1] > 1 else attention_mask
            sequence_lengths = adjusted_mask.sum(dim=1) - 1
            batch_size = attention_mask.shape[0]
            image_embed = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths + 1  # +1 because of prepended point cloud token
            ]
        else:
            image_embed = last_hidden_state[:, -1, :]

        # 4. Final fusion (cross-attention or direct use)
        if self.use_cross_attention and self.cross_attention is not None:
            fused_embeds = self.cross_attention(pc_context_embed, image_embed)
        else:
            # Use point cloud context token directly
            fused_embeds = pc_context_embed

        # 5. Multi-task regression predictions
        regression_outputs = self.regression_head(fused_embeds)

        # 6. Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(regression_outputs, labels)

        return Qwen3VLRegressionOutput(
            loss=loss,
            logits=regression_outputs,
            hidden_states=outputs.get('hidden_states'),
            attentions=outputs.get('attentions'),
            fused_embeddings=fused_embeds,
        )


class Qwen3VLRegressionLoader(Qwen3VLLoader):
    """
    Custom Model Loader for Qwen3-VL Multi-Task Regression with Early Fusion.

    Additional Args:
        task_weights: Weights for each task in loss computation (optional, can be string or list)
        point_cloud_dim: Dimension of point cloud features (default: 512)
        use_cross_attention: Whether to use cross-attention for final fusion (default: True)
    """

    def __init__(
        self,
        *args,
        task_weights: Optional[Union[str, List[float]]] = None,
        point_cloud_dim: int = 512,
        use_cross_attention: bool = True,
        **kwargs
    ):
        # Parse task_weights if it's a string
        if task_weights is not None and isinstance(task_weights, str):
            # Handle formats like "(1.0,1.0,1.0)" or "1.0,1.0,1.0" or "[1.0,1.0,1.0]"
            task_weights_str = task_weights.strip('()[]')
            task_weights = [float(w.strip()) for w in task_weights_str.split(',') if w.strip()]

        self.task_weights = task_weights
        self.point_cloud_dim = point_cloud_dim
        self.use_cross_attention = use_cross_attention
        super().__init__(*args, **kwargs)

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        """Load base model and wrap with early fusion regression components."""
        # Load base Qwen3-VL model
        base_model = super().get_model(model_dir, config, processor, model_kwargs)

        # Get model dimensions
        hidden_size = config.hidden_size
        dtype = base_model.dtype

        # Get num_regression_tasks from model_info.num_labels
        num_regression_tasks = self.model_info.num_labels
        if num_regression_tasks is None:
            raise ValueError(
                "num_labels must be specified for regression tasks. "
                "Please set --num_labels in your training arguments."
            )

        # Validate task_weights length matches num_regression_tasks
        if self.task_weights is not None:
            if len(self.task_weights) != num_regression_tasks:
                raise ValueError(
                    f"task_weights length ({len(self.task_weights)}) must match "
                    f"num_labels ({num_regression_tasks})"
                )

        # Wrap with early fusion regression components
        model = Qwen3VLForMultiTaskRegression(
            base_model=base_model,
            point_cloud_dim=self.point_cloud_dim,
            hidden_size=hidden_size,
            num_regression_tasks=num_regression_tasks,
            task_weights=self.task_weights,
            use_cross_attention=self.use_cross_attention,
            dtype=dtype
        )

        # model_meta is set by _postprocess_model, but we also set it here for access during model wrapping
        # Note: config, name_or_path, generation_config are accessed via properties that delegate to base_model
        model.model_meta = self.model_meta

        return model


# ============================================================================
# Model Registration
# ============================================================================

register_model(
    ModelMeta(
        'qwen3_vl_multi_regression',
        [
            ModelGroup([
                Model('Qwen/Qwen3-VL-Embedding-2B', 'Qwen/Qwen3-VL-Embedding-2B'),
                Model('Qwen/Qwen3-VL-Embedding-8B', 'Qwen/Qwen3-VL-Embedding-8B'),
            ]),
        ],
        Qwen3VLRegressionLoader,
        template=TemplateType.qwen3_vl_emb,
        model_arch=ModelArch.qwen3_vl,
        architectures=['Qwen3VLForConditionalGeneration'],
        requires=['transformers>=4.57', 'qwen_vl_utils>=0.0.14'],
        tags=['vision', 'regression', 'point-cloud', 'multi-task', 'early-fusion']
    )
)
