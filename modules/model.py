import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class ResBlock1dTF(nn.Module):
    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        self.block_t = nn.Sequential(
            nn.ReflectionPad1d(dilation * (kernel_size // 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, bias=False, dilation=dilation, groups=dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True)
        )
        self.block_f = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True)
        )
        self.shortcut = nn.Conv1d(dim, dim, 1, 1)

    def forward(self, x):
        return self.shortcut(x) + self.block_f(x) + self.block_t(x)


class OutputHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OutputHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)  # 第一个线性层,降维到一半
        self.fc2 = nn.Linear(input_dim // 2, num_classes)  # 第二个线性层,用于分类
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Dropout层,防止过拟合

    def forward(self, x):
        # x 的形状为 (sequence_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # 将维度顺序调整为 (batch_size, sequence_length, embedding_dim)
        x = torch.mean(x, dim=1)  # 在序列长度维度上进行平均池化
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: (N, B, E) = (50, 32, 512)
        x = x[0]  # 取cls token对应的输出 shape: (B, E) = (32, 512)
        x = self.fc(x)  # (B, nclass)
        return x


class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=512, n_layers=6, nhead=6, dim_feedforward=512):
        super(TAggregate, self).__init__()
        self.num_tokens = 1
        drop_rate = 0.1
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, activation="gelu", dim_feedforward=dim_feedforward, dropout=drop_rate)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.pos_embed = nn.Parameter(torch.zeros(50, embed_dim, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,  nn.Parameter):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()  # torch.Size([49, 32, 512]) --->>> torch.Size([49, 512, 32]) N,C,B
        # 将cls_token扩展到(1, 512, 32)的形状
        cls_tokens = self.cls_token.expand(-1, -1, x.shape[2])  # torch.Size([1, 1, 32]) -> torch.Size([1, 512, 32])
        # 将cls_tokens拼接到x的第一个维度上
        x = torch.cat((cls_tokens, x), dim=0)  # torch.Size([50, 512, 32])
        x += self.pos_embed
        x.transpose_(1, 0)  # -> torch.Size([512, 50, 32]) E,N,B
        x = x.permute(1, 2, 0)  # torch.Size([512, 50, 32]) -> torch.Size([50, 32, 512])
        o = self.transformer_enc(x)  # torch.Size([50, 32, 512])  # 需要的维度(N, B, E)
        return o


class AADownsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(AADownsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        ha = torch.arange(1, filt_size // 2 + 1 + 1, 1)
        a = torch.cat((ha, ha.flip(dims=[-1, ])[1:])).float()
        a = a / a.sum()
        filt = a[None, :]
        self.register_buffer('filt', filt[None, :, :].repeat((self.channels, 1, 1)))

    def forward(self, x):
        x_pad = F.pad(x, (self.filt_size // 2, self.filt_size // 2), "reflect")
        y = F.conv1d(x_pad, self.filt, stride=self.stride, padding=0, groups=x.shape[1])
        return y


class Down(nn.Module):
    def __init__(self, channels, d=2, k=3):
        super().__init__()
        kk = d + 1
        self.down = nn.Sequential(
            nn.ReflectionPad1d(kk // 2),
            nn.Conv1d(channels, channels * 2, kernel_size=kk, stride=1, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.LeakyReLU(0.2, True),
            AADownsample(channels=channels * 2, stride=d, filt_size=k)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class model(nn.Module):
    def __init__(self, nf=32, clip_length=None, embed_dim=128, n_layers=4, nhead=8, factors=[4, 4, 4, 4],
                 n_classes=None, dim_feedforward=512):
        super().__init__()
        self.start = nn.Sequential(
            nn.ReflectionPad2d((3, 3, 0, 0)),  # 在宽度方向填充3个元素，高度方向不填充
            nn.Conv2d(1, 16, kernel_size=(2, 7), stride=1, bias=False),  # 使用 2x7 的卷积核，输出通道数为1
            nn.BatchNorm2d(16),  # 输入通道数为1  # 需要形状：(N, C, H, W)
            nn.Conv2d(16, nf, kernel_size=(1, 1), stride=1, bias=False),  # 使用 1x1 的卷积核，将通道数转换为 nf
            nn.BatchNorm2d(nf),  # 输入通道数为 nf, default: 32
            nn.LeakyReLU(0.2, True)
        )

        factors = [4, 4, 4, 4]
        self.down = nn.Sequential(
            Down(channels=nf, d=factors[0], k=factors[0] * 2 + 1),
            ResBlock1dTF(dim=nf * 2, dilation=1, kernel_size=15),
            Down(channels=nf * 2, d=factors[1], k=factors[1] * 2 + 1),
            Down(channels=nf * 4, d=factors[2], k=factors[2] * 2 + 1),
            ResBlock1dTF(dim=nf * 8, dilation=1, kernel_size=15),
            Down(channels=nf * 8, d=factors[3], k=factors[3] * 2 + 1)
        )

        # 更新 nf 的值
        nf *= 16
        factors = [2, 2]
        # 创建 self.down2 模块
        self.down2 = nn.Sequential(
            ResBlock1dTF(dim=nf, dilation=1, kernel_size=15),
            ResBlock1dTF(dim=nf, dilation=3, kernel_size=15),
            ResBlock1dTF(dim=nf, dilation=9, kernel_size=15),
            Down(channels=nf, d=factors[0], k=factors[0] * 2 + 1),
            ResBlock1dTF(dim=nf * 2, dilation=1, kernel_size=15),
            ResBlock1dTF(dim=nf * 2, dilation=3, kernel_size=15),
            ResBlock1dTF(dim=nf * 2, dilation=9, kernel_size=15),
            Down(channels=nf * 2, d=factors[1], k=factors[1] * 2 + 1)
        )

        # self.project = nn.Conv1d(nf, embed_dim, 1)
        self.project = nn.Conv1d(in_channels=nf * 4, out_channels=embed_dim, kernel_size=1)
        self.clip_length = clip_length
        self.tf = TAggregate(embed_dim=embed_dim, clip_length=clip_length, n_layers=n_layers, nhead=nhead,
                             dim_feedforward=dim_feedforward)
        self.apply(self._init_weights)
        self.outputHead = ClassificationHead(input_dim=embed_dim, num_classes=n_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.start(x)  # torch.Size([32, 1, 2, 50000])  注释的形状是指=右边输入x的形状
        x = x.squeeze(2)  # torch.Size([32, 32 1, 50000])
        x = self.down(x)  # torch.Size([32, 32, 50000])
        x = self.down2(x)  # torch.Size([32, 512, 196])
        x = self.project(x)  # torch.Size([32, 2048, 49])
        # 转换形状以适应Transformer编码器(B, C, N) --->>>  (N, B, C)
        x = x.permute(2, 0, 1)  # torch.Size([32, 512, 49])
        x = self.tf(x)  # torch.Size([49, 32, 512])
        pred = self.outputHead(x)  # input x: torch.Size([50，32, 512])
        return pred  # torch.Size([32, N_classes])


if __name__ == '__main__':
    pass
