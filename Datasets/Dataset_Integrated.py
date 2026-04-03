import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from Datasets.load_vcf import load_labels_from_vcf
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class IntegratedDataset(Dataset):
    def __init__(self, mode='training'):
        # ================= 配置路径 =================
        self.root_feat = os.path.join(PROJECT_ROOT, "feature", "Biological")
        self.root_feat_new = os.path.join(PROJECT_ROOT, "feature", "Biological")
        self.root_physio_user = os.path.join(PROJECT_ROOT,"feature", "physicochemical", "COSMIC")
        self.root_hyena = os.path.join(PROJECT_ROOT, "feature", "hyena-dna")
        self.root_bert = os.path.join(PROJECT_ROOT, "feature", "DNAbert-2")
        self.root_shape_npy = os.path.join(PROJECT_ROOT, "Datasets", "processed_shape")
        self.vcf_path = os.path.join(PROJECT_ROOT, "data", f"COSMIC_{mode}.vcf")
        print(f"\n>>> 初始化 {mode} 集成数据集 <<<")
        # 1. 加载 Label
        self.labels = load_labels_from_vcf(self.vcf_path)
        self.num_samples = len(self.labels)

        # 2. 加载序列特征 (Sequence)
        print("  [Tabular] Loading sequence features...")
        feat_author_rest = self._load_tabular_from_list(mode)

        # 3. 加载Basic-1整合特征
        print("  [Tabular] Loading YOUR Basic Integrated features (BASIC-1.txt)...")
        feat_new_basic = self._load_new_basic_features(mode)

        # 4. 加载物化特征 (User Physicochemical)
        print("  [Tabular] Loading YOUR Physicochemical features...")
        feat_user_physio = self._load_user_physicochemical(mode)

        # 5. 加载 Hyena-DNA 特征
        print("  [Tabular] Loading Hyena-DNA features...")
        feat_hyena = self._load_hyena_dna(mode)

        # 6. 拼接所有表格类特征
        min_len_tab = min(len(feat_author_rest), len(feat_new_basic), len(feat_user_physio), len(feat_hyena))
        if not (len(feat_author_rest) == len(feat_new_basic) == len(feat_hyena)==len(feat_user_physio)):
            print(f"  Warning: Tabular lengths differ! Truncating to {min_len_tab}.")
            feat_author_rest = feat_author_rest[:min_len_tab]
            feat_new_basic = feat_new_basic[:min_len_tab]
            feat_user_physio = feat_user_physio[:min_len_tab]
            feat_hyena = feat_hyena[:min_len_tab]

        self.feat_tabular = np.concatenate([feat_author_rest, feat_new_basic, feat_user_physio, feat_hyena], axis=1)

        print(f"  [Tabular] Merged Shape: {self.feat_tabular.shape} "
              f"(AuthorRest: {feat_author_rest.shape[1]} + BasicNew: {feat_new_basic.shape[1]} + "
              f"UserPhysio: {feat_user_physio.shape[1]} + Hyena: {feat_hyena.shape[1]})")

        # 7. 加载 Shape
        shape_file = f"shape_all_{mode}.npy"
        shape_path = os.path.join(self.root_shape_npy, shape_file)
        if os.path.exists(shape_path):
            self.feat_shape = np.load(shape_path)
            if len(self.feat_shape) == self.num_samples + 1:
                self.feat_shape = self.feat_shape[1:]
        else:
            self.feat_shape = np.zeros((self.num_samples, 15, 100))

        # 8. 加载 DNABERT-2
        bert_file = "train_embeddings.npy" if mode == 'training' else "test_embeddings.npy"
        self.feat_bert = np.load(os.path.join(self.root_bert, bert_file))
        if self.feat_bert.ndim == 3 and self.feat_bert.shape[1] != 768:
            self.feat_bert = self.feat_bert.transpose(0, 2, 1)
        if len(self.feat_bert) == self.num_samples + 1:
            self.feat_bert = self.feat_bert[1:]

        # 9. 最终对齐
        min_len = min(self.num_samples, len(self.feat_tabular), len(self.feat_shape), len(self.feat_bert))
        if self.num_samples != min_len:
            print(f"  Final trimming: {self.num_samples} -> {min_len}")
            self.labels = self.labels[:min_len]
            self.feat_tabular = self.feat_tabular[:min_len]
            self.feat_shape = self.feat_shape[:min_len]
            self.feat_bert = self.feat_bert[:min_len]

        print("   数据加载完成！")
    def _load_new_basic_features(self, mode):
        filename = "train_BASIC-1.txt" if mode == 'training' else "test_BASIC-1.txt"
        file_path = os.path.join(self.root_feat_new, filename)

        if not os.path.exists(file_path):
            print(f"Error: Basic file not found {file_path}")
            return np.zeros((self.num_samples, 0), dtype=np.float32)

        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)

            # 根据你提供的 head 提取所有有效列
            cols_to_keep = [
                # --- Conservation (My Generated) ---
                'verPhCons', 'verPhyloP', 'mamPhCons', 'mamPhyloP',
                'priPhCons', 'priPhyloP', 'GerpS',

                # --- Epigenetics / Cell Line ---
                'Gm12878', 'H1hesc', 'Hepg2', 'Hmec', 'Hsmm',
                'Huvec', 'K562', 'Nhek', 'Nhlf',

                # --- Functional Scores ---
                'gdi', 'gdi_phred', 'rvis', 'lof_score',
                'SilVA', 'FATHMM-XF', 'PhD_SNPg_SCORE',
                'CADD_PHRED', 'CADD_Raw',
                'SpliceAI_max', 'GERP..',

                # --- MMS Splicing ---
                'MMS_delta_logit_psi', 'MMS_ref_acceptorIntron', 'MMS_alt_acceptorIntron',
                'MMS_ref_acceptor', 'MMS_alt_acceptor', 'MMS_ref_exon', 'MMS_alt_exon',
                'MMS_ref_donor', 'MMS_alt_donor', 'MMS_ref_donorIntron', 'MMS_alt_donorIntron',
                'MMS_pathogenicity', 'MMS_efficiency',

                # --- Deep Splicing / Structure ---
                'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL',
                'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',

                # --- Other Splicing/Regional ---
                'CpG.', 'SR.', 'SR..1', 'FAS6.', 'FAS6..1',
                'MES', 'dMES', 'MES.', 'MES..1',
                'MEC.MC.', 'MEC.CS.', 'MES.KM.',
                'PESE.', 'PESE..1', 'PESS.', 'PESS..1',

                # --- New/Extra ---
                'pdG_pre_50', 'pdG_post_50', 'pvar_pre_50', 'pvar_post_50'
            ]

            # 仅保留存在的列
            existing_cols = [c for c in cols_to_keep if c in df.columns]


            df_filtered = df[existing_cols]

            if len(df_filtered) > self.num_samples:
                df_filtered = df_filtered.iloc[:self.num_samples]

            data = df_filtered.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
            return data

        except Exception as e:
            print(f"Error loading Basic features: {e}")
            return np.zeros((self.num_samples, 0), dtype=np.float32)

    def _load_user_physicochemical(self, mode):
        prefix = "train" if mode == 'training' else "test"

        feature_types = [
            'diffe_CKSNAP_type_1', 'diffe_Kmer_type_1', 'diffe_Mismatch', 'diffe_MMI',
            'diffe_NAC', 'diffe_NMBroto', 'diffe_RCKmer_type_1',
            'diffe_Z_curve_9bit', 'diffe_Z_curve_12bit', 'diffe_Z_curve_36bit',
            'diffe_Z_curve_48bit', 'diffe_Z_curve_144bit',
            'mutation_CKSNAP_type_1', 'mutation_Kmer_type_1', 'mutation_Mismatch', 'mutation_MMI',
            'mutation_NAC', 'mutation_NMBroto', 'mutation_RCKmer_type_1',
            'mutation_Z_curve_9bit', 'mutation_Z_curve_12bit', 'mutation_Z_curve_36bit',
            'mutation_Z_curve_48bit', 'mutation_Z_curve_144bit',
            'normal_CKSNAP_type_1', 'normal_Kmer_type_1', 'normal_Mismatch', 'normal_MMI',
            'normal_NAC', 'normal_NMBroto', 'normal_RCKmer_type_1',
            'normal_Z_curve_9bit', 'normal_Z_curve_12bit', 'normal_Z_curve_36bit',
            'normal_Z_curve_48bit', 'normal_Z_curve_144bit'
        ]

        data_list = []
        total_dim = 0
        print(f"  [Debug] Checking User Physicochemical files ({mode})...")

        for f_type in feature_types:
            filename = f"{prefix}_{f_type}.txt"
            file_path = os.path.join(self.root_physio_user, filename)

            if not os.path.exists(file_path):
                print(f"    [X] Missing: {filename}")
                continue

            try:
                # === 自动检测分隔符 ===
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                sep = '\t' if '\t' in first_line else (',' if ',' in first_line else None)

                # === 读取数据 ===
                df = pd.read_csv(file_path, sep=sep, low_memory=False)

                # 截断样本数
                if len(df) > self.num_samples:
                    df = df.iloc[:self.num_samples]

                # === 转换并存储 ===
                data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
                data_list.append(data)

                # === 打印维度信息 (关键调试点) ===
                mark = "(!)" if data.shape[1] == 1 else ""
                print(f"    -> Loaded {filename:<35} | Shape: {data.shape} | Sep: {repr(sep)} {mark}")

                total_dim += data.shape[1]

            except Exception as e:
                print(f"    [Error] {filename}: {e}")

        if not data_list:
            print("    [Warning] No user physicochemical features loaded!")
            return np.zeros((self.num_samples, 0), dtype=np.float32)

        print(f"  [Debug] Total User Physio Dimension: {total_dim}")
        return np.concatenate(data_list, axis=1)

    def _load_tabular_from_list(self, mode):
        prefix = "training" if mode == 'training' else "test"
        file_path = os.path.join(self.root_feat, f"{prefix}_sequence.txt")
        if not os.path.exists(file_path):
            return np.zeros((self.num_samples, 0))
        try:
            with open(file_path, 'r') as f:
                sep = '\t' if '\t' in f.readline() else (',' if ',' in f.readline() else None)
            df = pd.read_csv(file_path, sep=sep, header=None, low_memory=False)
            if len(df) == self.num_samples + 1: df = df.iloc[1:]
            data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
            if data.ndim == 1: data = data.reshape(-1, 1)
            return data
        except Exception:
            return np.zeros((self.num_samples, 0))

    def _load_hyena_dna(self, mode):
        filename = f"hyena-dna_{mode}.txt"
        file_path = os.path.join(self.root_hyena, filename)
        if not os.path.exists(file_path):
            return np.zeros((self.num_samples, 256), dtype=np.float32)
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            if len(df) > self.num_samples: df = df.iloc[:self.num_samples]
            return df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
        except Exception:
            return np.zeros((self.num_samples, 256), dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.feat_shape[idx]),
                torch.tensor(self.feat_bert[idx]),
                torch.tensor(self.feat_tabular[idx]),
                torch.tensor(self.labels[idx]))
