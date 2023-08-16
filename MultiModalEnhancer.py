import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from Image_encoder import img_encoder


#Text Encoder
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma[:x.shape[-1]]
        beta = self.beta[:x.shape[-1]]
        return F.layer_norm(x, x.shape[1:], gamma.repeat(x.shape[1], 1), beta.repeat(x.shape[1], 1))
class Residual_1D(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block_1D(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_1D(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual_1D(out_channels, out_channels))
    return nn.Sequential(*blk)


class ResNet_1D(nn.Module):
    def __init__(self, input_channels=3, output_channels=512):
        super().__init__()
        self.net2_1 = nn.Sequential(nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm1d(64), nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.net2_2 = nn.Sequential(*resnet_block_1D(64, 64, 2, first_block=True))

        self.net2_3 = nn.Sequential(*resnet_block_1D(64, 128, 2))  # 每个模块使⽤2个残差块

        self.net2_4 = nn.Sequential(*resnet_block_1D(128, 256, 2))

        self.net2_5 = nn.Sequential(*resnet_block_1D(256, 512, 2))

        self.net2_6 = nn.Sequential(*resnet_block_1D(512, 1024, 2))

        self.net2_7 = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

        self.linear = nn.Sequential(nn.Linear(1024, 4096),nn.Dropout(0.5),
                                    nn.Linear(4096, output_channels))

    def forward(self, img):

        feature2_1 = self.net2_1(img)
        feature2_2 = self.net2_2(feature2_1)
        feature2_3 = self.net2_3(feature2_2)
        feature2_4 = self.net2_4(feature2_3)
        feature2_5 = self.net2_5(feature2_4)
        feature2_6 = self.net2_6(feature2_5)
        feature2_7 = self.net2_7(feature2_6)
        output = self.linear(feature2_7)
        return output

custom_resnet_1D = ResNet_1D(input_channels = 1, output_channels = 512)


#Image Encoder
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class ResNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=512):
        super().__init__()
        self.net2_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.net2_2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))

        self.net2_3 = nn.Sequential(*resnet_block(64, 128, 2))  # 每个模块使⽤2个残差块

        self.net2_4 = nn.Sequential(*resnet_block(128, 256, 2))

        self.net2_5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.net2_6 = nn.Sequential(*resnet_block(512, 1024, 2))

        self.net2_7 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        self.linear = nn.Sequential(nn.Linear(1024, 4096),nn.Dropout(0.5),
                                    nn.Linear(4096, 512))

    def forward(self, img):
        feature2_1 = self.net2_1(img)
        feature2_2 = self.net2_2(feature2_1)
        feature2_3 = self.net2_3(feature2_2)
        feature2_4 = self.net2_4(feature2_3)
        feature2_5 = self.net2_5(feature2_4)
        feature2_6 = self.net2_6(feature2_5)
        feature2_7 = self.net2_7(feature2_6)
        output = self.linear(feature2_7)

        return output

custom_resnet = ResNet(input_channels = 3, output_channels = 512)


#
class MultiModalEnhancer(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, num_heads=8, head_dim=64, num_tokens=1000):
        super().__init__()

        # self.image_resnet = custom_resnet      image Encoder 1
        self.image_encoder  = img_encoder     # image Encoder 2
        self.signal_resnet = custom_resnet_1D

        self.multi_modal_fusion = MultiModalFusionBlock(input_dim=512, num_heads=num_heads, head_dim=head_dim)
        self.feature_enhancement = FeatureEnhancementBlock(output_dim=512, num_heads=num_heads, head_dim=head_dim)

        self.fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(512 * 64, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_tokens)
        )

    def forward(self, signal, image):

        signal = rearrange(signal, 'c (r p) -> c r p', r=1)
        image_embed, _ = self.image_encoder(image)
        signal_embed = self.signal_resnet(signal)

        signal_embed = repeat(signal_embed, 'h w -> h w c', c=512)
        image_embed = repeat(image_embed, 'h w -> h w c', c=512)

        fused_output = self.multi_modal_fusion(image_embed, signal_embed)
        enhanced_output = self.feature_enhancement(fused_output)

        final_output = self.fc(enhanced_output)
        return final_output

# First Fusion
class MultiModalFusionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.norm = LayerNorm(input_dim)

        inner_dim = num_heads * head_dim

        self.image_projection = nn.Linear(input_dim, inner_dim)
        self.signal_projection = nn.Linear(input_dim, head_dim * 2)
        # ...

    def forward(self, image, signal):
        image = self.norm(image)
        signal = self.norm(signal)

        projected_image = self.image_projection(image)
        projected_image = rearrange(projected_image, 'b n (h d) -> b h n d', h=self.num_heads)
        projected_image = projected_image * self.scale

        projected_signal, value = self.signal_projection(signal).chunk(2, dim=-1)
        # ...

        similarity = einsum('b h i d, b j d -> b h i j', projected_image, projected_signal)
        similarity = similarity - similarity.amax(dim=-1, keepdim=True)
        attn = similarity.softmax(dim=-1)

        out = einsum('b h i j, b j d -> b h i d', attn, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # ...

        return out

#Feature Enhancement
class FeatureEnhancementBlock(nn.Module):
    def __init__(self, output_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.norm = LayerNorm(output_dim)

        self.residual_block = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, 1, 1),
            nn.Conv1d(output_dim, num_heads, 1, 1)
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, 1, 1)
        )
        self.relu = nn.ReLU()

    def forward(self, features):
        features = self.norm(features)

        residual = self.residual_block(features)

        residual = rearrange(residual, 'b h n -> b n h')

        enhanced_features = self.cnn(residual)
        enhanced_features = F.interpolate(enhanced_features, size=(64,), mode='linear', align_corners=False)
        features = F.interpolate(features, size=(64,), mode='linear', align_corners=False)
        final_features = self.relu(enhanced_features) + features # torch.Size([32, 512, 64])
        final_features = self.relu(final_features)

        return final_features

#Test
newnet = MultiModalEnhancer(input_dim=512, output_dim=512, num_heads=8, head_dim=64, num_tokens=1000)

x = torch.rand([32, 260])
img = torch.rand([32, 3, 256, 256])

out = newnet(signal = x, image = img)
print(out.shape)