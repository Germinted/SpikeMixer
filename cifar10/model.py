import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import surrogate
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = ['SpikeDynaMixer', 'spike_dynamixer_s', 'spike_dynamixer_m', 'spike_dynamixer_l']


# =============================================================================
# Core Components
# =============================================================================

class SpikeDynaMixerOp(nn.Module):
    """
    Spiking Dynamic Mixing Operation.

    Adapted from DynaMixer's DynaMixerOp:
    1. Compress: X @ W_compress -> reduced dimension (with BN + LIF)
    2. Generate: compressed @ W_generate -> L x L weight matrix
    3. Mix: weights @ X (dynamic token mixing)

    Key SNN adaptations:
    - BatchNorm + LIF after compression (instead of nothing)
    - No softmax on weights (not spike-compatible), use scaled weights
    - Multi-head structure preserved for efficiency

    Args:
        dim: Channel dimension C
        seq_len: Sequence length L (number of tokens along axis)
        num_head: Number of mixing heads
        reduced_dim: Reduced dimension per head (d << C)
    """

    def __init__(self, dim: int, seq_len: int, num_head: int = 8, reduced_dim: int = 2):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim

        # Compress: dim -> num_head * reduced_dim
        self.compress = nn.Linear(dim, num_head * reduced_dim, bias=False)
        self.compress_bn = nn.BatchNorm1d(num_head * reduced_dim)
        self.compress_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend='cupy'
        )

        # Generate: seq_len * reduced_dim -> seq_len * seq_len
        self.generate = nn.Linear(seq_len * reduced_dim, seq_len * seq_len, bias=False)

        # Output projection (like DynaMixer)
        self.out = nn.Linear(dim, dim, bias=False)
        self.out_bn = nn.BatchNorm1d(dim)

        # Scale factor for weight normalization (replaces softmax)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.compress.weight)
        nn.init.xavier_uniform_(self.generate.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor, T: int, B_actual: int) -> torch.Tensor:
        """
        Args:
            x: Input [T*B, L, C]
            T: Number of timesteps
            B_actual: Actual batch size
        Returns:
            y: Output [T*B, L, C]
        """
        TB, L, C = x.shape

        # Compress: [TB, L, C] -> [TB, L, num_head * reduced_dim]
        compressed = self.compress(x)

        # BN + LIF on compressed features
        # Reshape for BN: [TB*L, h*d] -> BN -> [T, B*L, h*d] for LIF
        compressed = self.compress_bn(compressed.reshape(-1, self.num_head * self.reduced_dim))
        compressed = compressed.reshape(T, B_actual * L, self.num_head * self.reduced_dim)
        compressed = self.compress_lif(compressed)
        compressed = compressed.reshape(TB, L, self.num_head * self.reduced_dim)

        # Reshape for weight generation: [TB, L, h, d] -> [TB, h, L, d] -> [TB, h, L*d]
        weights = compressed.reshape(TB, L, self.num_head, self.reduced_dim)
        weights = weights.permute(0, 2, 1, 3).reshape(TB, self.num_head, -1)

        # Generate: [TB, h, L*d] -> [TB, h, L*L] -> [TB, h, L, L]
        weights = self.generate(weights)
        weights = weights.reshape(TB, self.num_head, L, L)

        # Scale weights (instead of softmax, use learned scale for stability)
        weights = weights * self.scale

        # Multi-head mixing: [TB, L, h, C//h]
        x_heads = x.reshape(TB, L, self.num_head, C // self.num_head)
        x_heads = x_heads.permute(0, 2, 3, 1)  # [TB, h, C//h, L]

        # Mix: [TB, h, C//h, L] @ [TB, h, L, L] = [TB, h, C//h, L]
        mixed = torch.matmul(x_heads, weights)

        # Reshape back: [TB, h, C//h, L] -> [TB, L, C]
        mixed = mixed.permute(0, 3, 1, 2).reshape(TB, L, C)

        # Output projection with BN
        out = self.out(mixed)
        out = self.out_bn(out.reshape(-1, C)).reshape(TB, L, C)

        return out


class Reweight(nn.Module):
    """
    Reweight module for adaptive fusion of multiple branches.

    From DynaMixer: computes per-channel weights for H/W/C branch fusion.

    Args:
        dim: Channel dimension
        num_branches: Number of branches to fuse (default 3: H, W, C)
        reduction: Reduction ratio for hidden dimension
    """

    def __init__(self, dim: int, num_branches: int = 3, reduction: int = 4):
        super().__init__()
        self.dim = dim
        self.num_branches = num_branches
        hidden_dim = max(dim // reduction, 16)

        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim * num_branches, bias=False)
        self.gelu = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C] (global pooled features)
        Returns:
            weights: [num_branches, B, C, 1, 1] for broadcasting
        """
        B = x.shape[0]

        # MLP: [B, C] -> [B, hidden] -> [B, C * num_branches]
        a = self.fc1(x)
        a = self.gelu(a)
        a = self.fc2(a)

        # Reshape and softmax across branches
        a = a.reshape(B, self.dim, self.num_branches)
        a = a.permute(2, 0, 1)  # [num_branches, B, C]

        # Add spatial dims for broadcasting: [num_branches, B, C, 1, 1]
        a = a.unsqueeze(-1).unsqueeze(-1)

        return a


class SpikeDynaMixerBlock(nn.Module):
    """
    Spiking DynaMixer Block.

    Structure (following DynaMixer):
        h = mix_h(x)              # Axial H mixing
        w = mix_w(x)              # Axial W mixing
        c = mlp_c(x)              # Channel mixing
        a = reweight(h + w + c)   # Adaptive fusion weights
        y = h * a[0] + w * a[1] + c * a[2]
        output = x + LIF(BN(y))   # Residual with spiking

    Args:
        dim: Channel dimension
        resolution: Spatial resolution (H = W assumed)
        num_head: Number of heads for mixing
        reduced_dim: Reduced dimension per head
        drop_path: DropPath rate
    """

    def __init__(
        self,
        dim: int,
        resolution: int,
        num_head: int = 8,
        reduced_dim: int = 2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.num_head = num_head

        # Axial mixing: H and W directions
        self.mix_h = SpikeDynaMixerOp(dim, resolution, num_head, reduced_dim)
        self.mix_w = SpikeDynaMixerOp(dim, resolution, num_head, reduced_dim)

        # Channel mixing (simple linear like DynaMixer)
        self.mlp_c = nn.Linear(dim, dim, bias=False)
        self.mlp_c_bn = nn.BatchNorm1d(dim)
        self.mlp_c_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend='cupy'
        )

        # Reweight for 3 branches
        self.reweight = Reweight(dim, num_branches=3, reduction=4)

        # Output: BN -> LIF
        self.bn = nn.BatchNorm2d(dim)
        self.lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend='cupy'
        )

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [T, B, C, H, W]
        Returns:
            y: Output [T, B, C, H, W]
        """
        T, B, C, H, W = x.shape
        identity = x

        # Reshape for axial mixing: [T*B, H, W, C]
        x_hw = x.permute(0, 1, 3, 4, 2).reshape(T * B, H, W, C)

        # H mixing (along rows): [T*B, W, H, C] -> [T*B*W, H, C]
        x_h = x_hw.permute(0, 2, 1, 3).reshape(T * B * W, H, C)
        h = self.mix_h(x_h, T, B * W)
        h = h.reshape(T * B, W, H, C).permute(0, 2, 1, 3)  # [T*B, H, W, C]

        # W mixing (along columns): [T*B, H, W, C] -> [T*B*H, W, C]
        x_w = x_hw.reshape(T * B * H, W, C)
        w = self.mix_w(x_w, T, B * H)
        w = w.reshape(T * B, H, W, C)  # [T*B, H, W, C]

        # Channel mixing
        c = self.mlp_c(x_hw)
        c = self.mlp_c_bn(c.reshape(-1, C)).reshape(T, B * H * W, C)
        c = self.mlp_c_lif(c)
        c = c.reshape(T * B, H, W, C)  # [T*B, H, W, C]

        # Compute reweight factors from pooled features
        pooled = (h + w + c).mean(dim=(1, 2))  # [T*B, C]
        a = self.reweight(pooled)  # [3, T*B, C, 1, 1]

        # Reshape h, w, c for broadcasting: [T*B, C, H, W]
        h = h.permute(0, 3, 1, 2)
        w = w.permute(0, 3, 1, 2)
        c = c.permute(0, 3, 1, 2)

        # Weighted fusion
        y = h * a[0] + w * a[1] + c * a[2]  # [T*B, C, H, W]

        # Reshape to [T, B, C, H, W]
        y = y.reshape(T, B, C, H, W)

        # Output: BN -> LIF
        y = self.bn(y.flatten(0, 1)).reshape(T, B, C, H, W)
        y = self.lif(y)

        # Residual connection with DropPath
        return identity + self.drop_path(y)


class SPS(nn.Module):
    """
    Spiking Patch Splitting.

    Converts input images to spike-based patch embeddings.
    Uses progressive channel expansion with MaxPool for downsampling.

    Structure:
        Stage 1: Conv -> BN -> LIF (no pool)
        Stage 2: Conv -> BN -> LIF (no pool)
        Stage 3: Conv -> BN -> LIF -> MaxPool
        Stage 4: Conv -> BN -> LIF -> MaxPool
        RPE: Conv -> BN -> LIF (residual position encoding)

    Args:
        img_size_h, img_size_w: Input image size
        patch_size: Downsampling factor (typically 4)
        in_channels: Input channels (3 for RGB)
        embed_dims: Output embedding dimension
    """

    def __init__(
        self,
        img_size_h: int = 32,
        img_size_w: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dims: int = 384,
    ):
        super().__init__()
        self.image_size = (img_size_h, img_size_w)
        self.patch_size = patch_size
        self.H = img_size_h // patch_size
        self.W = img_size_w // patch_size
        self.num_patches = self.H * self.W

        # Determine pooling stages based on patch_size
        # patch_size=4 -> 2 pools (stages 3, 4)
        if patch_size == 4:
            self.pool_stages = [False, False, True, True]
        elif patch_size == 8:
            self.pool_stages = [False, True, True, True]
        elif patch_size == 16:
            self.pool_stages = [True, True, True, True]
        else:
            import math
            num_pools = int(math.log2(patch_size))
            self.pool_stages = [False] * (4 - num_pools) + [True] * num_pools

        # Channel progression: in_channels -> C/8 -> C/4 -> C/2 -> C
        ch = [in_channels, embed_dims // 8, embed_dims // 4, embed_dims // 2, embed_dims]

        # Stage 1
        self.proj_conv = nn.Conv2d(ch[0], ch[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(ch[1])
        self.proj_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend='cupy'
        )
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1) if self.pool_stages[0] else nn.Identity()

        # Stage 2
        self.conv1 = nn.Conv2d(ch[1], ch[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch[2])
        self.lif1 = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend='cupy'
        )
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1) if self.pool_stages[1] else nn.Identity()

        # Stage 3
        self.conv2 = nn.Conv2d(ch[2], ch[3], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch[3])
        self.lif2 = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend='cupy'
        )
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1) if self.pool_stages[2] else nn.Identity()

        # Stage 4
        self.conv3 = nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch[4])
        self.lif3 = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend='cupy'
        )
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1) if self.pool_stages[3] else nn.Identity()

        # Relative Position Encoding
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend='cupy'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [T, B, C, H, W]
        Returns:
            y: Output [T, B, embed_dims, H/patch_size, W/patch_size]
        """
        T, B, _, H, W = x.shape
        cur_h, cur_w = H, W

        # Stage 1
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, cur_h, cur_w)
        x = self.proj_lif(x).flatten(0, 1)
        x = self.pool0(x)
        if self.pool_stages[0]:
            cur_h, cur_w = cur_h // 2, cur_w // 2

        # Stage 2
        x = self.conv1(x)
        x = self.bn1(x).reshape(T, B, -1, cur_h, cur_w)
        x = self.lif1(x).flatten(0, 1)
        x = self.pool1(x)
        if self.pool_stages[1]:
            cur_h, cur_w = cur_h // 2, cur_w // 2

        # Stage 3
        x = self.conv2(x)
        x = self.bn2(x).reshape(T, B, -1, cur_h, cur_w)
        x = self.lif2(x).flatten(0, 1)
        x = self.pool2(x)
        if self.pool_stages[2]:
            cur_h, cur_w = cur_h // 2, cur_w // 2

        # Stage 4
        x = self.conv3(x)
        x = self.bn3(x).reshape(T, B, -1, cur_h, cur_w)
        x = self.lif3(x).flatten(0, 1)
        x = self.pool3(x)
        if self.pool_stages[3]:
            cur_h, cur_w = cur_h // 2, cur_w // 2

        # RPE with residual
        x_feat = x.reshape(T, B, -1, cur_h, cur_w)
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, cur_h, cur_w)
        x = self.rpe_lif(x)
        x = x + x_feat

        return x


class SpikeDynaMixer(nn.Module):
    """
    SpikeDynaMixer - SNN Version of DynaMixer.

    Architecture:
    1. SPS: Spiking Patch Splitting (multi-stage conv + MaxPool)
    2. SpikeDynaMixerBlocks: Axial mixing (H, W) + channel mixing with reweighting
    3. Classification head: Global pool -> Linear

    Key Features:
    - Dynamic mixing adapted for SNNs (no softmax)
    - Axial mixing reduces complexity from O(N^2) to O(N)
    - Adaptive branch fusion with learnable reweighting
    - Energy-efficient spike-based computation

    Args:
        img_size_h, img_size_w: Input image size
        patch_size: Downsampling factor
        in_channels: Input channels
        num_classes: Number of output classes
        embed_dims: Embedding dimension
        depths: Number of transformer blocks
        num_heads: Number of mixing heads
        reduced_dims: Reduced dimension per head
        drop_path_rate: Stochastic depth rate
        T: Number of timesteps
    """

    def __init__(
        self,
        img_size_h: int = 32,
        img_size_w: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dims: int = 384,
        depths: int = 4,
        num_heads: int = 8,
        reduced_dims: int = 2,
        drop_path_rate: float = 0.1,
        T: int = 4,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # Patch embedding
        self.patch_embed = SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
        )

        resolution = img_size_h // patch_size

        # SpikeDynaMixer blocks
        self.blocks = nn.ModuleList([
            SpikeDynaMixerBlock(
                dim=embed_dims,
                resolution=resolution,
                num_head=num_heads,
                reduced_dim=reduced_dims,
                drop_path=dpr[i],
            )
            for i in range(depths)
        ])

        self.head = nn.Linear(embed_dims, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = self.patch_embed(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, H, W]
        Returns:
            logits: [B, num_classes]
        """
        # Expand to T timesteps: [B, C, H, W] -> [T, B, C, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # Extract features
        x = self.forward_features(x)  # [T, B, C, H', W']

        # Global average pooling: [T, B, C, H, W] -> [T, B, C]
        x = x.mean(dim=(-2, -1))

        # Temporal average: [T, B, C] -> [B, C]
        x = x.mean(dim=0)

        # Classification
        x = self.head(x)

        return x


# =============================================================================
# Model Factory Functions
# =============================================================================

@register_model
def spikemixer_s(pretrained: bool = False, **kwargs) -> SpikeDynaMixer:
    """
    SpikeMixer Small - Efficiency-focused variant.

    Target metrics (CIFAR-10):
    - Params: ~3M
    - OPs: ~1.5G
    - Energy @25% FR: ~7mJ
    """
    _ = pretrained
    model = SpikeDynaMixer(
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dims=256,
        depths=4,
        num_heads=4,
        reduced_dims=2,
        drop_path_rate=0.1,
        T=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def spikemixer_m(pretrained: bool = False, **kwargs) -> SpikeDynaMixer:
    """
    SpikeMixer Medium - Balanced variant.

    Target metrics (CIFAR-10):
    - Params: ~6M
    - OPs: ~2.5G
    - Energy @25% FR: ~12mJ
    """
    _ = pretrained
    model = SpikeDynaMixer(
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dims=384,
        depths=4,
        num_heads=8,
        reduced_dims=2,
        drop_path_rate=0.15,
        T=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def spikemixer_l(pretrained: bool = False, **kwargs) -> SpikeDynaMixer:
    """
    SpikeMixer Large - Accuracy-focused variant.

    Target metrics (CIFAR-10):
    - Params: ~12M
    - OPs: ~4G
    - Energy @25% FR: ~18mJ
    """
    _ = pretrained
    model = SpikeDynaMixer(
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dims=512,
        depths=8,
        num_heads=16,
        reduced_dims=2,
        drop_path_rate=0.2,
        T=4,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model