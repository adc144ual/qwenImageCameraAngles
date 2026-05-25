import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    """Bloque Bottleneck para ResNet (usado en layer1 de HRNet)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
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
        return self.relu(out + residual)


class BasicBlock(nn.Module):
    """Bloque básico para ResNet (usado en stages 2-4 de HRNet)."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class HighResolutionModule(nn.Module):
    """Módulo multi-rama con fusión entre resoluciones."""

    def __init__(self, num_branches, num_channels, num_blocks, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, num_channels, num_blocks)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_idx, num_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(num_channels[branch_idx], num_channels[branch_idx]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, num_channels, num_blocks):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_channels, num_blocks))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                            ))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_channels[j], momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True),
                            ))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + nn.functional.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode='bilinear',
                        align_corners=True
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class PoseHRNet(nn.Module):
    """
    HRNet para estimación de pose humana.
    Configuración: W48 (COCO 17 keypoints).
    """

    def __init__(self, width=48, num_joints=17):
        super().__init__()
        C = width

        # Stem
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Layer1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, 1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Transition1
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, C, 3, padding=1, bias=False),
                nn.BatchNorm2d(C, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(256, C * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C * 2, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            HighResolutionModule(2, [C, C * 2], num_blocks=4),
        )

        # Transition2
        self.transition2 = nn.ModuleList([
            None,
            None,
            nn.Sequential(nn.Sequential(
                nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C * 4, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage3
        stage3_modules = []
        for i in range(4):
            stage3_modules.append(
                HighResolutionModule(3, [C, C * 2, C * 4], num_blocks=4,
                                     multi_scale_output=True)
            )
        self.stage3 = nn.Sequential(*stage3_modules)

        # Transition3
        self.transition3 = nn.ModuleList([
            None,
            None,
            None,
            nn.Sequential(nn.Sequential(
                nn.Conv2d(C * 4, C * 8, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C * 8, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage4
        stage4_modules = []
        for i in range(3):
            multi_scale_output = True if i < 2 else False
            stage4_modules.append(
                HighResolutionModule(4, [C, C * 2, C * 4, C * 8], num_blocks=4,
                                     multi_scale_output=multi_scale_output)
            )
        self.stage4 = nn.Sequential(*stage4_modules)

        # Final layer
        self.final_layer = nn.Conv2d(C, num_joints, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        # Transition1
        x_list = []
        for i in range(2):
            x_list.append(self.transition1[i](x))

        # Stage2
        y_list = self.stage2[0](x_list)

        # Transition2
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # Stage3
        y_list = x_list
        for module in self.stage3:
            y_list = module(y_list)

        # Transition3
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # Stage4
        y_list = x_list
        for module in self.stage4:
            y_list = module(y_list)

        # Output
        x = self.final_layer(y_list[0])
        return x


def load_hrnet_model(
    model_path: str,
    width: int = 48,
    num_joints: int = 17,
    device: torch.device = torch.device('cpu')
) -> PoseHRNet:
    """Carga HRNet pre-entrenado."""
    model = PoseHRNet(width=width, num_joints=num_joints)

    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model.to(device)

    logger.info(f"✓ HRNet cargado desde {model_path}")
    return model