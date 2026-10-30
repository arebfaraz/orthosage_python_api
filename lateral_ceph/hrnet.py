"""
HRNet (High-Resolution Net) for Cephalometric Landmark Detection.
Supports HRNet-W32 and HRNet-W48 variants.
Reference: Deep High-Resolution Representation Learning for Visual Recognition (Sun et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_ch, out_ch, stride=1, padding=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=padding, bias=False)


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def make_layer(block, inplanes, planes, num_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
        )
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


class HRModule(nn.Module):
    """
    One module of the High-Resolution stage: N parallel branches with fusion.
    """

    def __init__(self, num_branches, num_blocks, num_channels, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = nn.ModuleList([
            self._make_branch(num_blocks[i], num_channels[i])
            for i in range(num_branches)
        ])
        self.fuse_layers = self._make_fuse_layers(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def _make_branch(self, num_blocks, channels):
        layers = [BasicBlock(channels, channels) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def _make_fuse_layers(self, num_channels):
        if self.num_branches == 1:
            return None
        n_out = self.num_branches if self.multi_scale_output else 1
        fuse_layers = nn.ModuleList()
        for i in range(n_out):
            row = nn.ModuleList()
            for j in range(self.num_branches):
                if j > i:
                    # Lower res → higher res: 1x1 conv + BN + upsample
                    row.append(nn.Sequential(
                        conv1x1(num_channels[j], num_channels[i]),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest'),
                    ))
                elif j == i:
                    row.append(nn.Identity())
                else:
                    # Higher res → lower res: stride-2 conv chain
                    convs = []
                    for k in range(i - j):
                        in_ch = num_channels[j]
                        out_ch = num_channels[i] if k == i - j - 1 else num_channels[j]
                        convs += [
                            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM),
                        ]
                        if k < i - j - 1:
                            convs.append(nn.ReLU(inplace=True))
                    row.append(nn.Sequential(*convs))
            fuse_layers.append(row)
        return fuse_layers

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        if self.fuse_layers is None:
            return x

        x_fuse = []
        for i, row in enumerate(self.fuse_layers):
            y = sum(layer(x[j]) for j, layer in enumerate(row))
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(nn.Module):
    """
    HRNet for landmark heatmap regression.

    Args:
        num_landmarks: number of output heatmaps
        width: base channel width (32 → W32, 48 → W48)
    """

    STAGE_CFG = {
        2: {'num_modules': 1, 'num_blocks': [4, 4]},
        3: {'num_modules': 4, 'num_blocks': [4, 4, 4]},
        4: {'num_modules': 3, 'num_blocks': [4, 4, 4, 4]},
    }

    def __init__(self, num_landmarks=29, width=32):
        super().__init__()
        self.width = width
        C = width
        self.channels = [C, C * 2, C * 4, C * 8]

        # Stem: 2× stride-2 convolutions → 1/4 resolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 4 Bottleneck blocks at 1/4 resolution → 256ch
        self.stage1 = make_layer(Bottleneck, 64, 64, num_blocks=4)
        stage1_out = 64 * Bottleneck.expansion  # 256

        # Transition 1: create 2 branches [C, 2C]
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(stage1_out, self.channels[0], 3, padding=1, bias=False),
                nn.BatchNorm2d(self.channels[0], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(stage1_out, self.channels[1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.channels[1], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ),
        ])

        # Stage 2: 1 module, 2 branches
        self.stage2 = self._make_stage(2, self.channels[:2], multi_scale_output=True)

        # Transition 2: add 3rd branch [C*4]
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(self.channels[1], self.channels[2], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.channels[2], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ),
        ])

        # Stage 3: 4 modules, 3 branches
        self.stage3 = self._make_stage(3, self.channels[:3], multi_scale_output=True)

        # Transition 3: add 4th branch [C*8]
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(self.channels[2], self.channels[3], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.channels[3], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ),
        ])

        # Stage 4: 3 modules, 4 branches, only highest-res output
        self.stage4 = self._make_stage(4, self.channels[:4], multi_scale_output=False)

        # Detection head: 1x1 conv on highest-resolution branch
        self.head = nn.Conv2d(self.channels[0], num_landmarks, 1)

        self._init_weights()

    def _make_stage(self, stage_id, num_channels, multi_scale_output):
        cfg = self.STAGE_CFG[stage_id]
        num_modules = cfg['num_modules']
        num_blocks = cfg['num_blocks']
        n_branches = len(num_channels)
        modules = []
        for i in range(num_modules):
            # Only last module outputs all scales (if multi_scale_output=True)
            mso = multi_scale_output if i == num_modules - 1 else True
            modules.append(HRModule(n_branches, num_blocks, num_channels, multi_scale_output=mso))
        return nn.Sequential(*modules)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Detection head has no downstream activation (raw heatmap regression),
        # so the ReLU-tuned kaiming init above leaves it producing very large
        # initial values (observed: raw output in the hundreds against targets
        # in [0, 1]). Start it near zero instead so the first few optimizer
        # steps aren't fighting a huge initial diff — this is what triggers
        # NaN losses/gradients (worse on the MPS backend) early in training.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.head.bias)

    def _transition(self, x_list, transition, prev_branches):
        out = []
        for i, layer in enumerate(transition):
            if i < prev_branches and not isinstance(layer, nn.Identity):
                out.append(layer(x_list[i]))
            elif isinstance(layer, nn.Identity):
                out.append(x_list[i])
            else:
                # New branch: use last available branch as input
                out.append(layer(x_list[-1]))
        return out

    def forward(self, x):
        x = self.stem(x)            # (B, 64, H/4, W/4)
        x = self.stage1(x)          # (B, 256, H/4, W/4)

        # Transition 1: 256ch → [C, 2C] at [H/4, H/8]
        x = [self.transition1[0](x), self.transition1[1](x)]

        # Stage 2
        for m in self.stage2:
            x = m(x)

        # Transition 2: add branch at H/16
        x = [x[0], x[1], self.transition2[2](x[1])]

        # Stage 3
        for m in self.stage3:
            x = m(x)

        # Transition 3: add branch at H/32
        x = [x[0], x[1], x[2], self.transition3[3](x[2])]

        # Stage 4: returns only highest-resolution branch
        for m in self.stage4:
            x = m(x)

        # x is now a list with 1 element (multi_scale_output=False)
        feat = x[0]                 # (B, C, H/4, W/4)
        heatmaps = self.head(feat)  # (B, num_landmarks, H/4, W/4)
        return heatmaps


def build_hrnet(model_name='hrnet_w32', num_landmarks=29, pretrained=False):
    width_map = {'hrnet_w32': 32, 'hrnet_w48': 48}
    if model_name not in width_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(width_map)}")
    model = HRNet(num_landmarks=num_landmarks, width=width_map[model_name])

    if pretrained:
        _load_pretrained(model, model_name)

    return model


def _load_pretrained(model, model_name):
    """
    Load ImageNet pretrained weights from timm / HuggingFace.
    Tries up to 3 times with increasing timeouts before giving up gracefully.
    Skips the detection head (different output channels).
    """
    import os
    import timm

    # Increase HuggingFace download timeout (default is 10s – too short)
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')

    timm_names = {'hrnet_w32': 'hrnet_w32.ms_in1k', 'hrnet_w48': 'hrnet_w48.ms_in1k'}
    timm_name = timm_names.get(model_name, model_name)

    for attempt in range(1, 4):
        try:
            print(f"Downloading pretrained {timm_name} (attempt {attempt}/3)...")
            pretrained_model = timm.create_model(timm_name, pretrained=True)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()

            matched = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and 'head' not in k
                       and model_dict[k].shape == v.shape}
            model_dict.update(matched)
            model.load_state_dict(model_dict)
            print(f"Pretrained weights loaded: {len(matched)}/{len(model_dict)} layers matched.")
            return
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")

    print("Warning: pretrained weights could not be loaded – training from scratch.")
    print("  To load manually: download hrnet_w48_ms_in1k.pth and use --weights <path>")
