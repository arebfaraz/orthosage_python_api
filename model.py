"""HRNet-W32 pose-estimation architecture. Module names/shapes
(conv1/bn1/conv2/bn2/layer1/transitionN/stageN/final_layer) match the
cephalometric checkpoint's state dict exactly (verified with a strict
state_dict load) -- do not rename modules without re-verifying against
the checkpoint.
"""

import torch.nn as nn

BN_MOMENTUM = 0.1

NUM_LANDMARKS = 19
LANDMARK_LABELS = [
    "Sella (S)",
    "Nasion (N)",
    "Orbitale (Or)",
    "Porion (Po)",
    "A-Point (A)",
    "B-Point (B)",
    "Pogonion (Pog)",
    "Menton (Me)",
    "Gnathion (Gn)",
    "Gonion (Go)",
    "Lower Incisor Tip (L1)",
    "Upper Incisor Tip (U1)",
    "Upper Lip",
    "Lower Lip",
    "Subnasale (Sn)",
    "Soft Tissue Pogonion",
    "Posterior Nasal Spine (PNS)",
    "Anterior Nasal Spine (ANS)",
    "Articulare (Ar)",
]

HRNET_W32_CFG = {
    "NUM_JOINTS": NUM_LANDMARKS,
    "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "NUM_BLOCKS": [4], "NUM_CHANNELS": [64], "BLOCK": "BOTTLENECK"},
    "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [32, 64], "BLOCK": "BASIC"},
    "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [32, 64, 128], "BLOCK": "BASIC"},
    "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [32, 64, 128, 256], "BLOCK": "BASIC"},
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


BLOCKS_BY_NAME = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels, multi_scale_output=True):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches, num_inchannels = self.num_branches, self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        out_ch = num_inchannels[i] if last else num_inchannels[j]
                        layer = [
                            nn.Conv2d(num_inchannels[j], out_ch, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM),
                        ]
                        if not last:
                            layer.append(nn.ReLU(inplace=True))
                        conv3x3s.append(nn.Sequential(*layer))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y + x[j] if i == j else y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNetW32(nn.Module):
    """ResNet-style stem + 4-stage high-resolution backbone regressing a
    per-landmark heatmap (NUM_JOINTS, H/4, W/4)."""

    def __init__(self, cfg=HRNET_W32_CFG):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        s1 = cfg["STAGE1"]
        block = BLOCKS_BY_NAME[s1["BLOCK"]]
        self.layer1 = self._make_layer(block, 64, s1["NUM_CHANNELS"][0], s1["NUM_BLOCKS"][0])
        stage1_out_channel = block.expansion * s1["NUM_CHANNELS"][0]

        s2 = cfg["STAGE2"]
        block = BLOCKS_BY_NAME[s2["BLOCK"]]
        num_channels = [c * block.expansion for c in s2["NUM_CHANNELS"]]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(s2, num_channels)

        s3 = cfg["STAGE3"]
        block = BLOCKS_BY_NAME[s3["BLOCK"]]
        num_channels = [c * block.expansion for c in s3["NUM_CHANNELS"]]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(s3, num_channels)

        s4 = cfg["STAGE4"]
        block = BLOCKS_BY_NAME[s4["BLOCK"]]
        num_channels = [c * block.expansion for c in s4["NUM_CHANNELS"]]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(s4, num_channels, multi_scale_output=True)

        self.final_layer = nn.Conv2d(pre_stage_channels[0], cfg["NUM_JOINTS"], kernel_size=1)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    in_ch = num_channels_pre_layer[-1]
                    out_ch = num_channels_cur_layer[i] if j == i - num_branches_pre else in_ch
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = BLOCKS_BY_NAME[layer_config["BLOCK"]]

        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output or (i != num_modules - 1)
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = [self.transition1[i](x) if self.transition1[i] is not None else x for i in range(len(self.transition1))]
        y_list = self.stage2(x_list)

        x_list = [
            self.transition2[i](y_list[-1] if i >= len(y_list) else y_list[i]) if self.transition2[i] is not None
            else y_list[i]
            for i in range(len(self.transition2))
        ]
        y_list = self.stage3(x_list)

        x_list = [
            self.transition3[i](y_list[-1] if i >= len(y_list) else y_list[i]) if self.transition3[i] is not None
            else y_list[i]
            for i in range(len(self.transition3))
        ]
        y_list = self.stage4(x_list)

        return self.final_layer(y_list[0])
