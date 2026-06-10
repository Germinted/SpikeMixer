import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import surrogate
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import numpy as np

__all__ = ['SpikeMixerModel']

class MSN(nn.Module):
  """Multiplicative Spiking Neuron for 2D spatial data.

  Input: [T, B, C, H, W] or [TB, C, H, W] (auto-detected)
  Output: Same shape as input

  Gradient stability fixes:
  - Constrained gate scale
  - Input/output clamping
  - Bounded membrane shortcut
  - Smaller weight initialization
  """
  def __init__(self, in_channels=256, tau=2.0):
      super().__init__()
      self.in_channels = in_channels
      self.compress = nn.Conv2d(in_channels, in_channels, kernel_size=1)
      self.compress_bn = nn.BatchNorm2d(in_channels)

      # Initialize smaller
      nn.init.xavier_uniform_(self.compress.weight, gain=0.1)

      # Learnable gate scale (initialized smaller, constrained)
      self.gate_scale = nn.Parameter(torch.ones(1) * 0.2)

      # Pre-LIF normalization
      self.pre_lif_norm = nn.GroupNorm(min(32, in_channels), in_channels)

      self.lif = MultiStepLIFNode(
          tau=tau,
          detach_reset=True,
          backend='cupy',
          surrogate_function=surrogate.Sigmoid(alpha=4.0),
          v_threshold=1.0
      )

      # Membrane potential shortcut (constrained)
      self.mem_shortcut = nn.Parameter(torch.zeros(1))

  def forward(self, x: torch.Tensor):
      # Clamp input
      x = torch.clamp(x, -10.0, 10.0)

      if x.dim() == 5:
          T, B, C, H, W = x.shape
          x_flat = x.flatten(0, 1)
          dyna = self.compress(x_flat)
          dyna = self.compress_bn(dyna)
          dyna = dyna.reshape(T, B, C, H, W)

          # Clamp dyna before multiplication
          dyna = torch.clamp(dyna, -5.0, 5.0)

          # Scaled gating with constrained parameter
          gate_scale = torch.clamp(self.gate_scale, 0.01, 0.3)
          x_mod = gate_scale * (dyna * x) + x

          # Pre-LIF normalization
          x_mod = self.pre_lif_norm(x_mod.flatten(0, 1)).reshape(T, B, C, H, W)

          # LIF
          out = self.lif(x_mod)

          # Constrained membrane shortcut
          mem_shortcut = torch.clamp(self.mem_shortcut, -0.1, 0.1)
          out = out + mem_shortcut * torch.sigmoid(x_mod)

          return torch.clamp(out, -10.0, 10.0)
      else:
          dyna = self.compress(x)
          dyna = self.compress_bn(dyna)
          dyna = torch.clamp(dyna, -5.0, 5.0)
          gate_scale = torch.clamp(self.gate_scale, 0.01, 0.3)
          out = gate_scale * (dyna * x) + x
          return torch.clamp(out, -10.0, 10.0)
  
class MSN1d(nn.Module):
  """Multiplicative Spiking Neuron for 1D sequence data.

  Input: [T, B, C, L] or [B, C, L] (auto-detected)
  Output: Same shape as input

  Gradient stability fixes:
  - Constrained gate scale
  - Input/output clamping
  - Bounded membrane shortcut
  """
  def __init__(self, in_channels=8, tau=2.0):
      super().__init__()
      self.in_channels = in_channels
      self.compress = nn.Conv1d(in_channels, in_channels, kernel_size=1)
      self.compress_bn = nn.BatchNorm1d(in_channels)

      # Initialize smaller
      nn.init.xavier_uniform_(self.compress.weight, gain=0.1)

      # Learnable gate scale (constrained)
      self.gate_scale = nn.Parameter(torch.ones(1) * 0.2)

      # Pre-LIF normalization
      self.pre_lif_norm = nn.GroupNorm(min(32, in_channels), in_channels)

      self.lif = MultiStepLIFNode(
          tau=tau,
          detach_reset=True,
          backend='cupy',
          surrogate_function=surrogate.Sigmoid(alpha=4.0),
          v_threshold=1.0
      )

      # Membrane potential shortcut (constrained)
      self.mem_shortcut = nn.Parameter(torch.zeros(1))

  def forward(self, x: torch.Tensor):
      # Clamp input
      x = torch.clamp(x, -10.0, 10.0)

      if x.dim() == 4:
          T, B, C, L = x.shape
          x_flat = x.flatten(0, 1)
          dyna = self.compress(x_flat)
          dyna = self.compress_bn(dyna)
          dyna = dyna.reshape(T, B, C, L)

          # Clamp before multiplication
          dyna = torch.clamp(dyna, -5.0, 5.0)

          # Constrained gating
          gate_scale = torch.clamp(self.gate_scale, 0.01, 0.3)
          x_mod = gate_scale * (dyna * x) + x

          # Pre-LIF normalization
          x_mod = self.pre_lif_norm(x_mod.flatten(0, 1)).reshape(T, B, C, L)

          # LIF
          out = self.lif(x_mod)

          # Constrained membrane shortcut
          mem_shortcut = torch.clamp(self.mem_shortcut, -0.1, 0.1)
          out = out + mem_shortcut * torch.sigmoid(x_mod)

          return torch.clamp(out, -10.0, 10.0)
      else:
          dyna = self.compress(x)
          dyna = self.compress_bn(dyna)
          dyna = torch.clamp(dyna, -5.0, 5.0)
          gate_scale = torch.clamp(self.gate_scale, 0.01, 0.3)
          out = gate_scale * (dyna * x) + x
          return torch.clamp(out, -10.0, 10.0)

class DynaMixerOp(nn.Module):
    """Dynamic Mixer with stable weight generation.

    Input: [B, L, C] where B can be T*batch for temporal processing
    Output: [B, L, C]

    Gradient stability fixes:
    - Pre-initialized generate layer (no dynamic creation)
    - Scaled weight generation with clamping
    - No softmax (causes overflow) - use normalized linear weights
    """
    def __init__(self, dim, seq_len, num_head, L, reduced_dim=2):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim
        self.head_dim = dim // num_head

        self.norm = nn.LayerNorm(dim)
        self.compress = nn.Linear(dim, num_head * reduced_dim)

        # Pre-initialize generate layer (CRITICAL: avoid dynamic creation)
        self.generate = nn.Linear(seq_len * reduced_dim, seq_len * seq_len)
        nn.init.xavier_uniform_(self.generate.weight, gain=0.1)  # Small init
        if self.generate.bias is not None:
            nn.init.zeros_(self.generate.bias)

        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)

        # Learnable scale with constraint
        self.weight_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        B, L, C = x.shape

        # Clamp input for numerical stability
        x = torch.clamp(x, -10.0, 10.0)

        # Pre-normalize
        x_norm = self.norm(x)

        # Compress: [B, L, C] -> [B, L, num_head * reduced_dim]
        weights = self.compress(x_norm)

        # Reshape for weight generation: [B, num_head, L * reduced_dim]
        weights = weights.reshape(B, L, self.num_head, self.reduced_dim)
        weights = weights.permute(0, 2, 1, 3).reshape(B, self.num_head, L * self.reduced_dim)

        # Generate mixing weights with clamping (avoid overflow)
        weights = self.generate(weights)
        weights = torch.clamp(weights, -5.0, 5.0)  # Prevent extreme values
        weights = weights.reshape(B, self.num_head, L, L)

        # Normalize weights (stable alternative to softmax)
        # Use L2 normalization instead of softmax to avoid exp overflow
        weights = weights / (weights.norm(dim=-1, keepdim=True) + 1e-6)
        weights = weights * torch.clamp(self.weight_scale, 0.01, 1.0)

        # Multi-head mixing
        x_heads = x.reshape(B, L, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        # Mix with clamped weights
        mixed = torch.matmul(weights, x_heads)

        # Reshape back: [B, L, C]
        mixed = mixed.permute(0, 2, 1, 3).reshape(B, L, C)

        # Output projection with dropout
        return self.out(self.dropout(mixed))

class MLP(nn.Module):
    """Spiking MLP with gradient stability optimizations.

    Gradient stability fixes:
    - Constrained learnable scales
    - Input/output clamping
    - Smaller initialization for conv weights
    - Bounded skip weight
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 2

        # First layer
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(
            tau=2.0,  # More stable tau
            detach_reset=True,
            backend='cupy',
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            v_threshold=1.0
        )

        # Second layer
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(
            tau=2.0,
            detach_reset=True,
            backend='cupy',
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            v_threshold=1.0
        )

        # Initialize conv weights smaller for stability
        nn.init.xavier_uniform_(self.fc1_conv.weight, gain=0.1)
        nn.init.xavier_uniform_(self.fc2_conv.weight, gain=0.1)

        # Learnable layer scaling (initialized smaller)
        self.layer1_scale = nn.Parameter(torch.ones(1) * 0.3)
        self.layer2_scale = nn.Parameter(torch.ones(1) * 0.3)

        # Skip connection weight (initialized to zero, constrained)
        self.skip_weight = nn.Parameter(torch.zeros(1))
        self.skip_proj = nn.Conv1d(in_features, out_features, kernel_size=1) if in_features != out_features else nn.Identity()
        if not isinstance(self.skip_proj, nn.Identity):
            nn.init.xavier_uniform_(self.skip_proj.weight, gain=0.1)

        self.dropout = nn.Dropout(drop)
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, N = x.shape
        identity = x

        # Clamp input
        x = torch.clamp(x, -10.0, 10.0)

        # First layer with constrained scaling
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        layer1_scale = torch.clamp(self.layer1_scale, 0.01, 0.5)
        x = layer1_scale * self.fc1_lif(x)

        # Clamp and dropout
        x = torch.clamp(x, -10.0, 10.0)
        x = self.dropout(x)

        # Second layer with constrained scaling
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, self.c_output, N).contiguous()
        layer2_scale = torch.clamp(self.layer2_scale, 0.01, 0.5)
        x = layer2_scale * self.fc2_lif(x)

        # Skip connection with constrained weight
        skip_weight = torch.clamp(self.skip_weight, -0.1, 0.1)
        if isinstance(self.skip_proj, nn.Identity):
            skip = identity
        else:
            skip = self.skip_proj(identity.flatten(0, 1)).reshape(T, B, self.c_output, N)

        out = x + skip_weight * skip

        # Final output clamping
        return torch.clamp(out, -10.0, 10.0)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SSA(nn.Module):
    """Spiking Self-Attention with axial mixing and temporal processing.

    Input: [T, B, C, N] where N = H * W spatial patches
    Output: [T, B, C, N]

    Gradient stability fixes:
    - Constrained learnable scales (clamped to [0.01, 1.0])
    - Input/output clamping for numerical stability
    - Pre-initialized temporal mixer (no dynamic creation)
    - Bounded membrane shortcut
    """
    def __init__(self, dim, num_heads=8, seq_len=8, reduced_dim=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, max_T=16):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len

        # Axial mixers for H and W dimensions
        self.mix_h = DynaMixerOp(dim, seq_len, num_heads, seq_len, reduced_dim)
        self.mix_w = DynaMixerOp(dim, seq_len, num_heads, seq_len, reduced_dim)

        # Learnable scales for axial mixing (initialized smaller for stability)
        self.h_scale = nn.Parameter(torch.ones(1) * 0.3)
        self.w_scale = nn.Parameter(torch.ones(1) * 0.3)
        self.t_scale = nn.Parameter(torch.ones(1) * 0.3)

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Channel MLP with residual connection (smaller init)
        self.mlp_c = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        # Initialize MLP weights smaller
        for m in self.mlp_c.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.mlp_c_scale = nn.Parameter(torch.ones(1) * 0.3)

        # Pre-initialize temporal mixer (CRITICAL: avoid dynamic creation)
        self.max_T = max_T
        self.mlp_t = nn.Linear(max_T, max_T, bias=False)
        nn.init.eye_(self.mlp_t.weight)  # Identity initialization

        # Output projection with spiking
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend='cupy',
            surrogate_function=surrogate.Sigmoid(alpha=4.0)
        )

        # Membrane potential shortcut (initialized to zero, constrained)
        self.mem_shortcut = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(proj_drop)

    def forward(self, x):
        T, B, C, N = x.shape
        H = W = int(np.sqrt(N))

        # Clamp input for numerical stability
        x = torch.clamp(x, -10.0, 10.0)

        # Pre-normalization
        x_flat = x.permute(0, 1, 3, 2).reshape(T * B, N, C)
        x_norm = self.norm1(x_flat)

        # Temporal mixing with constrained scale
        t_scale = torch.clamp(self.t_scale, 0.01, 0.5)
        x_for_t = x_norm.reshape(T, B, N, C).permute(1, 2, 3, 0)  # [B, N, C, T]

        # Pad or truncate to max_T for consistent temporal mixing
        if T < self.max_T:
            pad = torch.zeros(B, N, C, self.max_T - T, device=x.device, dtype=x.dtype)
            x_for_t_padded = torch.cat([x_for_t, pad], dim=-1)
        else:
            x_for_t_padded = x_for_t[..., :self.max_T]

        x_t_mixed = self.mlp_t(x_for_t_padded)

        # Extract only the needed timesteps
        if T < self.max_T:
            x_t_mixed = x_t_mixed[..., :T]
        else:
            # If T > max_T, use identity for extra timesteps
            x_t_mixed = torch.cat([x_t_mixed, x_for_t[..., self.max_T:]], dim=-1)

        x_t = t_scale * x_t_mixed + (1 - t_scale) * x_for_t
        x_t = x_t.permute(3, 0, 1, 2)  # [T, B, N, C]

        # Reshape for spatial mixing
        x_spatial = x_t.reshape(T * B, N, C)

        # Axial H mixing with constrained scale
        h_scale = torch.clamp(self.h_scale, 0.01, 0.5)
        x_h = x_spatial.reshape(T * B, H, W, C).permute(0, 2, 1, 3).reshape(T * B * W, H, C)
        h_out = self.mix_h(x_h)
        h_out = h_out.reshape(T * B, W, H, C).permute(0, 2, 1, 3).reshape(T * B, N, C)

        # Axial W mixing with constrained scale
        w_scale = torch.clamp(self.w_scale, 0.01, 0.5)
        x_w = x_spatial.reshape(T * B, H, W, C).reshape(T * B * H, W, C)
        w_out = self.mix_w(x_w)
        w_out = w_out.reshape(T * B, H, W, C).reshape(T * B, N, C)

        # Gated combination with constrained scales
        combined = h_scale * h_out + w_scale * w_out

        # Channel mixing with constrained scale
        mlp_c_scale = torch.clamp(self.mlp_c_scale, 0.01, 0.5)
        c_out = mlp_c_scale * self.mlp_c(combined) + combined

        # Clamp intermediate values
        c_out = torch.clamp(c_out, -10.0, 10.0)

        # Output projection with BN and LIF
        c_out_pre = c_out.transpose(1, 2)
        c_out_bn = self.proj_bn(c_out_pre)
        c_out_reshape = c_out_bn.reshape(T, B, C, N)

        # LIF with constrained membrane shortcut
        spike_out = self.proj_lif(c_out_reshape)
        mem_shortcut = torch.clamp(self.mem_shortcut, -0.1, 0.1)
        c_out = spike_out + mem_shortcut * torch.sigmoid(c_out_reshape)

        # Final normalization
        c_flat = c_out.permute(0, 1, 3, 2).reshape(T * B, N, C)
        c_norm = self.norm2(c_flat)
        out = c_norm.reshape(T, B, N, C).permute(0, 1, 3, 2)

        # Final output clamping
        out = torch.clamp(out, -10.0, 10.0)

        return self.dropout(out)

class Block(nn.Module):
    """Transformer block with SSA and MLP.

    Input/Output: [T, B, C, N]

    Gradient stability fixes:
    - Constrained residual scales
    - Constrained gamma parameters
    - Input/output clamping
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        # Pre-normalization for stable gradients
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # SSA with attention
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Learnable residual scales (smaller initialization)
        self.scale1 = nn.Parameter(torch.ones(1) * 0.1)
        self.scale2 = nn.Parameter(torch.ones(1) * 0.1)

        # Layer-wise scaling (initialized to 1, will be constrained)
        self.gamma1 = nn.Parameter(torch.ones(dim))
        self.gamma2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        T, B, C, N = x.shape

        # Clamp input
        x = torch.clamp(x, -10.0, 10.0)

        # Pre-norm for attention path
        x_norm = self.norm1(x.permute(0, 1, 3, 2).reshape(T * B, N, C))
        x_norm = x_norm.reshape(T, B, N, C).permute(0, 1, 3, 2)

        # SSA with constrained residual scale
        attn_out = self.attn(x_norm)
        gamma1 = torch.clamp(self.gamma1, 0.1, 2.0)  # Constrain gamma
        attn_out = attn_out * gamma1.view(1, 1, -1, 1)
        scale1 = torch.clamp(self.scale1, 0.01, 0.3)  # Constrain scale
        x = x + scale1 * self.drop_path(attn_out)

        # Clamp intermediate
        x = torch.clamp(x, -10.0, 10.0)

        # Pre-norm for MLP path
        x_norm = self.norm2(x.permute(0, 1, 3, 2).reshape(T * B, N, C))
        x_norm = x_norm.reshape(T, B, N, C).permute(0, 1, 3, 2)

        # MLP with constrained residual scale
        mlp_out = self.mlp(x_norm)
        gamma2 = torch.clamp(self.gamma2, 0.1, 2.0)
        mlp_out = mlp_out * gamma2.view(1, 1, -1, 1)
        scale2 = torch.clamp(self.scale2, 0.01, 0.3)
        x = x + scale2 * self.drop_path(mlp_out)

        # Final output clamping
        return torch.clamp(x, -10.0, 10.0)


class SPS(nn.Module):
    """Spiking Patch Splitting with gradient flow optimizations.

    Progressive 4-stage embedding: 128x128 -> 8x8 (16x downsample)
    Channels: in_ch -> C/8 -> C/4 -> C/2 -> C

    Gradient flow improvements:
    - Sigmoid surrogate functions for smoother gradients
    - Learnable stage scales for gradient modulation
    - Skip connections between stages
    """
    def __init__(self, img_size_h=32, img_size_w=32, patch_size=4, in_channels=3, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        # Stage 1: Progressive embedding with smoother gradients
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(
            tau=1.5,  # Lower tau for early layers
            detach_reset=True,
            backend='cupy',
            surrogate_function=surrogate.Sigmoid(alpha=4.0),  # Smoother gradient
            v_threshold=1.0
        )
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # Stage 2
        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(
            tau=1.8,
            detach_reset=True,
            backend='cupy',
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            v_threshold=1.0
        )
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # Stage 3
        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(
            tau=2.2,
            detach_reset=True,
            backend='cupy',
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            v_threshold=1.0
        )
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # Stage 4
        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(
            tau=2.5,  # Higher tau for deeper layers
            detach_reset=True,
            backend='cupy',
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            v_threshold=1.0
        )
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # Learnable stage scales for gradient flow control
        self.stage_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.8),
            nn.Parameter(torch.ones(1) * 0.8),
            nn.Parameter(torch.ones(1) * 0.8),
            nn.Parameter(torch.ones(1) * 0.8),
        ])

        # Relative position encoding with improved design
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(
            tau=2.0,
            detach_reset=True,
            backend='cupy',
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            v_threshold=1.0
        )

        # Learnable RPE scale
        self.rpe_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        T, B, _, H, W = x.shape

        # Stage 1: Initial projection with learnable scale
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.stage_scales[0] * self.proj_lif(x)
        x = x.flatten(0, 1).contiguous()
        x = self.maxpool(x)

        # Stage 2: Feature refinement with learnable scale
        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.stage_scales[1] * self.proj_lif1(x)
        x = x.flatten(0, 1).contiguous()
        x = self.maxpool1(x)

        # Stage 3: Deep feature extraction with learnable scale
        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.stage_scales[2] * self.proj_lif2(x)
        x = x.flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        # Stage 4: Final embedding with learnable scale
        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//8, W//8).contiguous()
        x = self.stage_scales[3] * self.proj_lif3(x)
        x = x.flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        # Relative position encoding with learnable scale
        x_rpe = self.rpe_conv(x)
        x_rpe = self.rpe_bn(x_rpe).reshape(T, B, -1, H//16, W//16).contiguous()
        x_rpe = self.rpe_lif(x_rpe)
        x_rpe = x_rpe.flatten(0, 1)

        # Apply residual connection with learnable scaling
        x = x + self.rpe_scale * x_rpe

        # Apply dropout for regularization
        x = self.dropout(x)

        # Final reshape
        x = x.reshape(T, B, -1, (H//16)*(W//16)).contiguous()

        return x

class SpikeMixer(nn.Module):
    def __init__(self,
                 img_size_h=32, img_size_w=32, patch_size=4, in_channels=3, num_classes=100,
                 embed_dims=256, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=4, sr_ratios=1
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Improved stochastic depth decay rule with higher regularization
        dpr = [x.item() for x in torch.linspace(0, max(drop_path_rate, 0.1), depths)]

        # Patch embedding with better design
        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)
        num_patches = patch_embed.num_patches

        # Positional embedding with improved initialization
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))

        # Main transformer blocks with improved architecture
        block = nn.ModuleList([Block(
            dim=embed_dims,
            num_heads=num_heads,
            mlp_ratio=mlp_ratios,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[j],
            norm_layer=norm_layer,
            sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"pos_embed", pos_embed)
        setattr(self, f"block", block)

        # Final normalization layer
        self.norm = norm_layer(embed_dims)

        # Improved classification head with dropout
        self.pre_head_dropout = nn.Dropout(0.2)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize position embedding
        pos_embed = getattr(self, f"pos_embed")
        trunc_normal_(pos_embed, std=.02)

        # Apply improved initialization
        self.apply(self._init_weights)

        # Additional weight initialization for stability
        self._init_additional_weights()

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _init_additional_weights(self):
        # Initialize residual scaling parameters
        for m in self.modules():
            if hasattr(m, 'scale1'):
                nn.init.constant_(m.scale1, 0.1)
            if hasattr(m, 'scale2'):
                nn.init.constant_(m.scale2, 0.1)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        # Patch embedding
        x = patch_embed(x)

        # Add positional embedding
        pos_embed = getattr(self, f"pos_embed")
        if pos_embed is not None:
            T, B, C, N = x.shape
            # Ensure pos_embed matches the actual number of patches N
            if pos_embed.size(1) != N:
                # Interpolate pos_embed to match the actual number of patches
                pos_embed = F.interpolate(
                    pos_embed.reshape(1, int(np.sqrt(pos_embed.size(1))), int(np.sqrt(pos_embed.size(1))), C).permute(0, 3, 1, 2),
                    size=(int(np.sqrt(N)), int(np.sqrt(N))),
                    mode="bilinear"
                ).reshape(1, C, N).permute(0, 2, 1)

            # Expand pos_embed to match batch and time dimensions
            pos = pos_embed.unsqueeze(0).expand(T, B, -1, -1).transpose(-1, -2)
            x = x + pos

        # Apply transformer blocks
        for blk in block:
            x = blk(x)

        # Final pooling - average over spatial patches
        return x.mean(dim=3)  # [T, B, C, N] -> [T, B, C]

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] DVS input or [B, C, H, W] static image
        Returns:
            logits: [B, num_classes]
        """
        # Handle different input formats
        if x.dim() == 4:  # Static image [B, C, H, W]
            x = x.unsqueeze(1).repeat(1, 4, 1, 1, 1)  # -> [B, T=4, C, H, W]

        # Permute to [T, B, C, H, W] for SNN processing
        x = x.permute(1, 0, 2, 3, 4)

        # Extract features
        x = self.forward_features(x)  # -> [T, B, C]

        # Apply final normalization and dropout
        x = self.norm(x)
        x = self.pre_head_dropout(x)

        # Temporal averaging: [T, B, C] -> [B, C]
        x = x.mean(0)

        # Classification
        return self.head(x)


@register_model
def SpikeMixerModel(pretrained=False, **kwargs):
    model = SpikeMixer(
        # Improved configuration for better accuracy
        img_size_h=128,  # CIFAR-100 image height
        img_size_w=128,  # CIFAR-100 image width
        patch_size=16,  # 32/4 = 8 patches per dimension, reasonable for CIFAR-100
        embed_dims=512,  # Higher embedding dimension (single value for simplicity)
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,  # RGB images for CIFAR-100
        num_classes=10,
        qkv_bias=True,  # Enable bias for better representation
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=2,  # Increased depth for better representation
        sr_ratios=1,
        drop_rate=0.1,  # Add dropout
        attn_drop_rate=0.1,  # Add attention dropout
        drop_path_rate=0.1,  # Add stochastic depth
        **kwargs
    )
    model.default_cfg = _cfg()
    return model