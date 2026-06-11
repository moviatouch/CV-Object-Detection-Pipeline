import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("vending_pipeline.osnet")

# -----------------------------------------------------------------------------
# OSNet Core Building Blocks
# -----------------------------------------------------------------------------


class ConvLayer(nn.Module):
    """
    Generic convolution block:
    Conv2D -> BatchNorm -> Optional ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        relu=True
    ):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x


class LightConv3x3(nn.Module):
    """
    Lightweight 3x3 convolution using depthwise separable strategy.

    Official OSNet sequence:
    1x1 Pointwise Conv -> 3x3 Depthwise Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)

        return self.relu(x)


class ChannelGate(nn.Module):
    """
    Channel attention gate (Squeeze-and-Excitation style).
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelGate, self).__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True
        )

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            in_channels,
            kernel_size=1,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        identity = x

        x = self.global_avgpool(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return identity * x


class OSBlock(nn.Module):
    """
    Omni-Scale Feature Learning Block.

    This implementation closely follows the official OSNet architecture:
    - Multi-scale lightweight convolution branches
    - Branch-wise channel gating
    - Residual connection with final ReLU
    """

    def __init__(self, in_channels, out_channels, reduction=4):
        super(OSBlock, self).__init__()

        mid_channels = out_channels // reduction

        # ---------------------------------------------------------------------
        # Initial channel reduction
        # ---------------------------------------------------------------------

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size=1
        )

        # ---------------------------------------------------------------------
        # Multi-scale convolution branches
        # ---------------------------------------------------------------------

        self.conv2a = LightConv3x3(
            mid_channels,
            mid_channels
        )

        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels)
        )

        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels)
        )

        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels)
        )

        # ---------------------------------------------------------------------
        # Branch-specific channel gates
        # ---------------------------------------------------------------------

        self.gate_a = ChannelGate(mid_channels)
        self.gate_b = ChannelGate(mid_channels)
        self.gate_c = ChannelGate(mid_channels)
        self.gate_d = ChannelGate(mid_channels)

        # ---------------------------------------------------------------------
        # Final feature projection
        # Official OSNet uses projection without activation
        # ---------------------------------------------------------------------

        self.conv3 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size=1,
            relu=False
        )

        # ---------------------------------------------------------------------
        # Residual projection (if dimensions differ)
        # ---------------------------------------------------------------------

        self.downsample = None

        if in_channels != out_channels:

            self.downsample = ConvLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                relu=False
            )

    def forward(self, x):

        identity = x

        # Initial reduction
        x1 = self.conv1(x)

        # Multi-scale feature extraction
        x2 = (
            self.gate_a(self.conv2a(x1)) +
            self.gate_b(self.conv2b(x1)) +
            self.gate_c(self.conv2c(x1)) +
            self.gate_d(self.conv2d(x1))
        )

        # Final projection
        x3 = self.conv3(x2)

        # Residual projection if needed
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Official OSNet residual activation
        return F.relu(x3 + identity)


# -----------------------------------------------------------------------------
# OSNet Main Architecture
# -----------------------------------------------------------------------------


class OSNet(nn.Module):
    """
    OSNet (Omni-Scale Network) for Person/Object Re-Identification.

    Configured for the standard OSNet x1.0 architecture.
    """

    def __init__(self, num_classes=1000, loss='softmax'):
        super(OSNet, self).__init__()

        self.loss = loss

        # ---------------------------------------------------------------------
        # Stem
        # ---------------------------------------------------------------------

        self.conv1 = ConvLayer(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # ---------------------------------------------------------------------
        # Stage 2
        # ---------------------------------------------------------------------

        self.conv2 = nn.Sequential(
            OSBlock(64, 256),
            OSBlock(256, 256),
            ConvLayer(256, 256, kernel_size=1)
        )

        self.pool2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # ---------------------------------------------------------------------
        # Stage 3
        # ---------------------------------------------------------------------

        self.conv3 = nn.Sequential(
            OSBlock(256, 384),
            OSBlock(384, 384),
            ConvLayer(384, 384, kernel_size=1)
        )

        self.pool3 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # ---------------------------------------------------------------------
        # Stage 4
        # ---------------------------------------------------------------------

        self.conv4 = nn.Sequential(
            OSBlock(384, 512),
            OSBlock(512, 512),
            ConvLayer(512, 512, kernel_size=1)
        )

        # ---------------------------------------------------------------------
        # Head
        # ---------------------------------------------------------------------

        self.conv5 = ConvLayer(
            512,
            512,
            kernel_size=1
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Classification layer
        # Typically unused during inference-only ReID
        self.fc = nn.Linear(512, num_classes)

        self._init_params()

    def _init_params(self):
        """
        Initialize model parameters.
        """

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass for feature extraction.

        Returns:
            Tensor of shape [B, 512]
        """

        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.conv5(x)

        x = self.global_avgpool(x)

        embeddings = x.view(x.size(0), -1)

        return embeddings


# -----------------------------------------------------------------------------
# Model Builder
# -----------------------------------------------------------------------------


def osnet_x1_0(pretrained_path=None):
    """
    Build OSNet x1.0 model and optionally load pretrained weights.

    Args:
        pretrained_path (str | Path):
            Path to pretrained checkpoint.

    Returns:
        OSNet model instance.
    """

    model = OSNet(num_classes=1000)

    if pretrained_path and Path(pretrained_path).exists():

        LOGGER.info(f"Loading pretrained OSNet weights from: {pretrained_path}")

        state_dict = torch.load(
            pretrained_path,
            map_location='cpu'
        )

        # ---------------------------------------------------------------------
        # Extract nested state_dict if needed
        # ---------------------------------------------------------------------

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        elif 'model' in state_dict:
            state_dict = state_dict['model']

        model_dict = model.state_dict()

        filtered_state_dict = {}

        loaded_layers = 0
        skipped_layers = 0

        # ---------------------------------------------------------------------
        # Load only matching layers
        # ---------------------------------------------------------------------

        for k, v in state_dict.items():

            if k in model_dict and v.shape == model_dict[k].shape:

                filtered_state_dict[k] = v
                loaded_layers += 1

            else:

                skipped_layers += 1

                LOGGER.warning(
                    f"Skipping layer due to mismatch: {k}"
                )

        # ---------------------------------------------------------------------
        # Load compatible weights
        # ---------------------------------------------------------------------

        model.load_state_dict(
            filtered_state_dict,
            strict=False
        )

        LOGGER.info(
            f"OSNet weights loaded successfully. "
            f"Loaded: {loaded_layers}, "
            f"Skipped: {skipped_layers}"
        )

    else:

        LOGGER.warning(
            "No pretrained weights found. "
            "Using randomly initialized model."
        )

    return model