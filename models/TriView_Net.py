# models/TriView_Net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 模块 1: HDRNet 风格的 BERT 处理模块
# ==========================================
class MultiscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiscaleBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        out = torch.cat([c1, c3, c5], dim=1)
        return self.relu(self.bn(out))


class HDRNet_Encoder(nn.Module):
    def __init__(self, in_dim=768, base_dim=128):
        super(HDRNet_Encoder, self).__init__()
        # 1. 降维
        self.entry = nn.Sequential(
            nn.Conv1d(in_dim, base_dim, kernel_size=1),
            nn.BatchNorm1d(base_dim),
            nn.ReLU()
        )
        # 2. 多尺度特征提取
        # out_channels=64 -> total=192
        self.multiscale = MultiscaleBlock(base_dim, 64)

        # 3. DPCNN
        self.dpcnn = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.entry(x)
        x = self.multiscale(x)
        x = self.dpcnn(x)
        return self.pool(x).flatten(1)  # (B, 256)


# ==========================================
# 模块 2: DeepSTF 风格的 Shape/Seq 处理模块
# ==========================================
class Spatial_Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Spatial_Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)


class DeepSTF_Encoder(nn.Module):
    def __init__(self, in_channels=19, conv_dim=64, lstm_hidden=64, dropout=0.3):
        super(DeepSTF_Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.attn = Spatial_Attention(lstm_hidden * 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        feat = self.attn(lstm_out)
        return feat


# ==========================================
# 模块 3: TriView 融合网络 (主模型)
# ==========================================
class TriView_Net(nn.Module):
    def __init__(self, config):
        """
        config: 包含模型结构的配置字典
        """
        super(TriView_Net, self).__init__()

        # 解析配置
        bert_dim = config['bert_in_dim']
        spatial_dim = config['spatial_in_dim']
        num_experts = config['num_experts']
        dropout_rate = config['fusion_dropout']

        self.hdr_branch = HDRNet_Encoder(in_dim=bert_dim)  # Out: 256
        self.stf_branch = DeepSTF_Encoder(in_channels=spatial_dim)  # Out: 128 (64*2)

        # Meta特征处理
        self.meta_fc = nn.Sequential(
            nn.Linear(num_experts, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 融合层
        # 256(BERT) + 128(Shape) + 64(Meta) = 448
        fusion_input_dim = 256 + 128 + 64

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, spatial, bert, meta):
        f_stf = self.stf_branch(spatial)
        f_hdr = self.hdr_branch(bert)
        f_meta = self.meta_fc(meta)

        combined = torch.cat([f_hdr, f_stf, f_meta], dim=1)
        return self.classifier(combined).squeeze(1)