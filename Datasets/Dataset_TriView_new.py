# Datasets/Dataset_TriView_new.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from Datasets.load_vcf import load_labels_from_vcf


class TriViewDataset(Dataset):
    # 【修改点1】这里增加了 config=None 参数
    def __init__(self, mode='training', config=None):

        # ================== 1. 路径配置 ==================

        # 【修改点2】优先从 config 读取 Meta 路径，否则使用默认值
        if config and 'PATH_META' in config:
            self.root_meta = config['PATH_META']
        else:
            self.root_meta = "Datasets/processed_meta"  # 默认回退路径

        # 其他路径保持不变 (或者你也可以把它们放进 config)
        self.root_shape_npy = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR/Datasets/processed_shape/"
        self.root_onehot = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/one-hot/"
        self.root_bert = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAbert-2"

        # VCF 路径
        if config and 'PATH_VCF_DIR' in config:
            self.vcf_path = os.path.join(config['PATH_VCF_DIR'], f"COSMIC_{mode}.vcf")
        else:
            self.vcf_path = f"/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC/COSMIC_{mode}.vcf"

        print(f"\n>>> 初始化 TriViewDataset [{mode}] <<<")
        print(f"    Meta Source: {self.root_meta}")  # 打印确认路径是否正确

        # ================== 2. 加载数据 ==================

        # (A) Labels
        self.labels = load_labels_from_vcf(self.vcf_path)
        self.num_samples = len(self.labels)

        # (B) Meta Features (5 Experts)
        meta_file = f"meta_probs_5_{mode}.npy"
        meta_path = os.path.join(self.root_meta, meta_file)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta feature file not found: {meta_path}")
        self.feat_meta = np.load(meta_path)

        # (C) Spatial Features = OneHot(4) + Shape(15) -> Total 19

        # --- C1. One-Hot ---
        onehot_file = f"onehot_{mode}.npy"
        seq = np.load(os.path.join(self.root_onehot, onehot_file))
        # 维度修正: (N, 100, 4) -> (N, 4, 100)
        if seq.ndim == 3 and seq.shape[2] == 4:
            seq = seq.transpose(0, 2, 1)

        # --- C2. Shape (15通道) ---
        shape_file = f"shape_all_{mode}.npy"
        shp = np.load(os.path.join(self.root_shape_npy, shape_file))

        # --- C3. 长度对齐 (Target: 100bp) ---
        target_len = 100

        # 处理 Seq
        if seq.shape[2] > target_len:
            seq = seq[:, :, :target_len]
        elif seq.shape[2] < target_len:
            pad = target_len - seq.shape[2]
            seq = np.pad(seq, ((0, 0), (0, 0), (0, pad)), 'constant')

        # 处理 Shape
        if shp.shape[2] > target_len:
            shp = shp[:, :, :target_len]
        elif shp.shape[2] < target_len:
            pad = target_len - shp.shape[2]
            shp = np.pad(shp, ((0, 0), (0, 0), (0, pad)), 'constant')

        # --- C4. 拼接 ---
        min_spatial = min(len(seq), len(shp))
        self.feat_spatial = np.concatenate([seq[:min_spatial], shp[:min_spatial]], axis=1)

        # (D) BERT Features
        bert_file = f"dnabert2_{mode}_3d.npy"
        self.feat_bert = np.load(os.path.join(self.root_bert, bert_file))

        # ================== 3. 全局对齐 ==================
        final_len = min(self.num_samples, len(self.feat_meta), len(self.feat_spatial), len(self.feat_bert))

        self.labels = self.labels[:final_len]
        self.feat_meta = self.feat_meta[:final_len]
        self.feat_spatial = self.feat_spatial[:final_len]
        self.feat_bert = self.feat_bert[:final_len]

        print(f"  [Info] Samples Loaded: {final_len}")
        print(f"  [Info] Spatial Shape: {self.feat_spatial.shape} (Expect: N, 19, 100)")
        print(f"  [Info] Meta Shape:    {self.feat_meta.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.feat_spatial[idx], dtype=torch.float32),
                torch.tensor(self.feat_bert[idx], dtype=torch.float32),
                torch.tensor(self.feat_meta[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))