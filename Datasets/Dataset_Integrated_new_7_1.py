# Dataset_Integrated_new7-5.py将还是使用原作者的sequence,但是加入了我提取的其他特征，即也就是剔除了我的sequence。
# 7-5，替换了作者物化特征数据
#7-6还是使用作者的物化数据，替换score

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from Datasets.load_vcf import load_labels_from_vcf


class IntegratedDataset(Dataset):
    def __init__(self, mode='training'):
        # ================= 配置路径 =================
        self.root_feat = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC"
        self.root_feat_new = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/basic-new/COSMIC"

        # 1. 【恢复】取消注释，指向你的物化特征路径
        self.root_physio_user = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/COSMIC"

        self.root_hyena = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/hyena-dna"
        self.root_bert = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAbert-2"
        self.root_shape_npy = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR/Datasets/processed_shape/"
        self.vcf_path = f"/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC/COSMIC_{mode}.vcf"

        print(f"\n>>> 初始化 {mode} 集成数据集 (Model 7-5: User Physio Replaces Author Physio) <<<")

        # 1. 加载 Label
        self.labels = load_labels_from_vcf(self.vcf_path)
        self.num_samples = len(self.labels)

        # 2. 加载作者残留特征
        print("  [Tabular] Loading Author's remaining features (Sequence & Score)...")
        feat_author_rest = self._load_tabular_from_list(mode)

        # 3. 加载你的 Basic 整合特征
        print("  [Tabular] Loading YOUR Basic Integrated features (BASIC-1.txt)...")
        feat_new_basic = self._load_new_basic_features(mode)

        # 4. 【恢复】加载你生成的物化特征 (User Physicochemical)
        print("  [Tabular] Loading YOUR Physicochemical features...")
        feat_user_physio = self._load_user_physicochemical(mode)

        # 5. 加载 Hyena-DNA 特征
        print("  [Tabular] Loading Hyena-DNA features...")
        feat_hyena = self._load_hyena_dna(mode)

        # 6. 拼接所有表格类特征
        # 维度对齐检查 (加入 feat_user_physio)
        min_len_tab = min(len(feat_author_rest), len(feat_new_basic), len(feat_user_physio), len(feat_hyena))
        #min_len_tab = min(len(feat_author_rest), len(feat_new_basic), len(feat_hyena))

        if not (len(feat_author_rest) == len(feat_new_basic) == len(feat_hyena)==len(feat_user_physio)):
            print(f"  Warning: Tabular lengths differ! Truncating to {min_len_tab}.")
            feat_author_rest = feat_author_rest[:min_len_tab]
            feat_new_basic = feat_new_basic[:min_len_tab]
            feat_user_physio = feat_user_physio[:min_len_tab]  # 截断你的物化
            feat_hyena = feat_hyena[:min_len_tab]

        # 拼接顺序: [作者(Seq/Score), 你的Basic, 你的Physio, Hyena]
        self.feat_tabular = np.concatenate([feat_author_rest, feat_new_basic, feat_user_physio, feat_hyena], axis=1)
        #self.feat_tabular = np.concatenate([feat_author_rest, feat_new_basic, feat_hyena], axis=1)

        print(f"  [Tabular] Merged Shape: {self.feat_tabular.shape} "
              f"(AuthorRest: {feat_author_rest.shape[1]} + BasicNew: {feat_new_basic.shape[1]} + "
              #f" Hyena: {feat_hyena.shape[1]}")
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
        """
        加载整合后的 Basic 特征 (包含 Conservation, Score, Sequence, Splicing)
        对应文件: train_BASIC-1.txt / test_BASIC-1.txt
        """
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

                # # --- Sequence & Composition ---
                # 'DSP', 'RSCU', 'dRSCU', 'CpG?', 'CpG_exon',
                # 'f_premrna', 'f_mrna', 'TFBs', 'GC_Content',
                # 'CpG_Count', 'RSCU_alt',

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

            # 【可选】打印缺失的列以供调试
            # missing_cols = set(cols_to_keep) - set(existing_cols)
            # if missing_cols: print(f"Warning: Missing Basic columns: {missing_cols}")

            df_filtered = df[existing_cols]

            if len(df_filtered) > self.num_samples:
                df_filtered = df_filtered.iloc[:self.num_samples]

            data = df_filtered.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
            return data

        except Exception as e:
            print(f"Error loading Basic features: {e}")
            return np.zeros((self.num_samples, 0), dtype=np.float32)

    def _load_user_physicochemical(self, mode):
        """
        【增强版】加载你自己生成的物化特征
        新增功能：自动检测分隔符 + 打印详细维度日志
        """
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
                # 如果是1维，可能是误读，标红显示
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
        """
        加载作者原有的特征，但【排除】了已被 Basic-1 替代的特征
        (conservation, sequence, score, splicing 被移除)
        只保留作者的物化特征，方便与你的新物化特征共存或对比。
        """
        feat_names = [
            # === 以下四类已注释，因为在 _load_new_basic_features 中加载 ===
            # 'conservation',
            'sequence',
             #'score',
            # 'splicing',

            # # === 保留作者的物化特征 (如果想完全替换，可以将下面也注释掉) ===
            # 'diffe_feature2mer', 'diffe_featureCKSNAP', 'diffe_featureMismatch', 'diffe_featureNAC',
            # 'diffe_featureRC2mer', 'diffe_featureMMI', 'diffe_featureZ_curve_9bit',
            # 'diffe_featureZ_curve_12bit', 'diffe_featureZ_curve_36bit', 'diffe_featureZ_curve_48bit',
            # 'diffe_featureZ_curve_144bit', 'diffe_featureNMBroto',
            # 'mutation_2mer', 'mutation_CKSNAP', 'mutation_Mismatch', 'mutation_NAC', 'mutation_RC2mer',
            # 'mutation_MMI', 'mutation_Z_curve_9bit', 'mutation_Z_curve_12bit',
            # 'mutation_Z_curve_36bit', 'mutation_Z_curve_48bit', 'mutation_Z_curve_144bit',
            # 'mutation_NMBroto',
            # 'normal_2mer', 'normal_CKSNAP', 'normal_Mismatch', 'normal_NAC',
            # 'normal_RC2mer', 'normal_MMI', 'normal_Z_curve_9bit',
            # 'normal_Z_curve_12bit', 'normal_Z_curve_36bit', 'normal_Z_curve_48bit',
            # 'normal_Z_curve_144bit', 'normal_NMBroto'
         ]

        data_list = []
        for feat in feat_names:
            filename = f"{feat}_{mode}.txt"
            file_path = os.path.join(self.root_feat, filename)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r') as f:
                    line = f.readline()
                    sep = '\t' if '\t' in line else (',' if ',' in line else None)

                df = pd.read_csv(file_path, sep=sep, header=None, low_memory=False)

                if len(df) == self.num_samples + 1:
                    df = df.iloc[1:]
                elif len(df) != self.num_samples:
                    continue

                data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
                if data.ndim == 1: data = data.reshape(-1, 1)
                data_list.append(data)

            except Exception:
                pass

        if not data_list:
            return np.zeros((self.num_samples, 0))

        combined = np.concatenate(data_list, axis=1)
        return combined

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



# # Dataset_Integrated_new7-3.py将splice替换成我自己的
# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import numpy as np
# import os
# from Datasets.load_vcf import load_labels_from_vcf
#
#
# class IntegratedDataset(Dataset):
#     def __init__(self, mode='training'):
#         # ================= 配置路径 =================
#         self.root_feat = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC"
#         # 新特征 Basic-New 的根目录
#         self.root_feat_new = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/basic-new/COSMIC"
#         # 【新增】Hyena-DNA 特征根目录
#         self.root_hyena = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/hyena-dna"
#
#         self.root_bert = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAbert-2"
#         self.root_shape_npy = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR/Datasets/processed_shape/"
#         self.vcf_path = f"/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC/COSMIC_{mode}.vcf"
#
#         print(f"\n>>> 初始化 {mode} 集成数据集 (Model 7-1: Hyena Replace Embedding2) <<<")
#
#         # 1. 加载 Label
#         self.labels = load_labels_from_vcf(self.vcf_path)
#         self.num_samples = len(self.labels)
#
#         # 2. 加载原有的 Tabular 特征 (已移除 embedding2)
#         print("  [Tabular] Loading original legacy features (excluding embedding2)...")
#         feat_original = self._load_tabular_from_list(mode)
#
#         # 3. 加载 Basic-New 补充特征
#         print("  [Tabular] Loading NEW basic features...")
#         feat_new = self._load_new_basic_features(mode)
#
#         # 4. 【新增】加载 Hyena-DNA 特征
#         print("  [Tabular] Loading Hyena-DNA features...")
#         feat_hyena = self._load_hyena_dna(mode)
#
#         # 5. 拼接所有表格类特征
#         # 维度检查与截断：确保行数一致 (以最短的为准)
#         min_len_tab = min(len(feat_original), len(feat_new), len(feat_hyena))
#
#         if not (len(feat_original) == len(feat_new) == len(feat_hyena)):
#             print(f"  Warning: Tabular lengths differ! "
#                   f"Orig: {len(feat_original)}, New: {len(feat_new)}, Hyena: {len(feat_hyena)}. "
#                   f"Truncating to {min_len_tab}.")
#             feat_original = feat_original[:min_len_tab]
#             feat_new = feat_new[:min_len_tab]
#             feat_hyena = feat_hyena[:min_len_tab]
#
#         # 拼接: Original + BasicNew + Hyena
#         self.feat_tabular = np.concatenate([feat_original, feat_new, feat_hyena], axis=1)
#
#         print(f"  [Tabular] Merged Shape: {self.feat_tabular.shape} "
#               f"(Orig: {feat_original.shape[1]} + New: {feat_new.shape[1]} + Hyena: {feat_hyena.shape[1]})")
#
#         # 6. 加载 Shape
#         shape_file = f"shape_all_{mode}.npy"
#         shape_path = os.path.join(self.root_shape_npy, shape_file)
#
#         if os.path.exists(shape_path):
#             self.feat_shape = np.load(shape_path)
#             if len(self.feat_shape) == self.num_samples + 1:
#                 self.feat_shape = self.feat_shape[1:]
#         else:
#             print(f"Warning: Shape file not found at {shape_path}. Using zeros.")
#             self.feat_shape = np.zeros((self.num_samples, 15, 100))
#
#         # 7. 加载 DNABERT-2
#         if mode == 'training':
#             bert_file = "train_embeddings.npy"
#         else:
#             bert_file = "test_embeddings.npy"
#
#         self.feat_bert = np.load(os.path.join(self.root_bert, bert_file))
#         if self.feat_bert.ndim == 3 and self.feat_bert.shape[1] != 768:
#             self.feat_bert = self.feat_bert.transpose(0, 2, 1)
#         if len(self.feat_bert) == self.num_samples + 1:
#             self.feat_bert = self.feat_bert[1:]
#
#         # 8. 最终一致性检查
#         min_len = min(self.num_samples, len(self.feat_tabular), len(self.feat_shape), len(self.feat_bert))
#         if self.num_samples != min_len:
#             print(f"  最终长度对齐: {self.num_samples} -> {min_len}")
#             self.labels = self.labels[:min_len]
#             self.feat_tabular = self.feat_tabular[:min_len]
#             self.feat_shape = self.feat_shape[:min_len]
#             self.feat_bert = self.feat_bert[:min_len]
#
#         print("   数据加载与对齐完成！")
#
#     def _load_hyena_dna(self, mode):
#         """
#         加载 Hyena-DNA 特征代替 embedding2
#         文件格式: Header存在, Tab分隔, 256维
#         """
#         filename = f"hyena-dna_{mode}.txt"
#         file_path = os.path.join(self.root_hyena, filename)
#
#         if not os.path.exists(file_path):
#             print(f"Error: Hyena file not found at {file_path}")
#             return np.zeros((self.num_samples, 256), dtype=np.float32)
#
#         try:
#             # 读取数据，假设第一行是 Header (hyena-dna_0 ... hyena-dna_255)
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#
#             # 简单校验列数，如果是256维
#             if df.shape[1] < 256:
#                 print(f"Warning: Hyena feature dimension {df.shape[1]} < 256?")
#
#             # 长度处理：通常如果带 Header，read_csv 读取后的长度应正好等于样本数
#             # 如果文件末尾有空行或者多余行，进行截断
#             if len(df) > self.num_samples:
#                 df = df.iloc[:self.num_samples]
#
#             data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
#             return data
#
#         except Exception as e:
#             print(f"Error loading Hyena features: {e}")
#             return np.zeros((self.num_samples, 256), dtype=np.float32)
#
#     def _load_new_basic_features(self, mode):
#         """加载新的 Basic 特征 (包含由你自己生成的 Conservation 特征)"""
#         if mode == 'training':
#             filename = "train_BASIC-1.txt"
#         else:
#             filename = "test_BASIC-1.txt"
#
#         file_path = os.path.join(self.root_feat_new, filename)
#         if not os.path.exists(file_path):
#             # 注意：如果文件不存在，返回空会导致后续维度对齐出问题，但在你的环境中文件是存在的
#             return np.zeros((self.num_samples, 0), dtype=np.float32)
#
#         try:
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#
#             cols_to_keep = [
#                 # --- 【新增】你自己生成的 Conservation 特征 (7个) ---
#                 'verPhCons', 'verPhyloP', 'mamPhCons', 'mamPhyloP',
#                 'priPhCons', 'priPhyloP', 'GerpS',
#
#                 # --- 原有的 Basic 特征 ---
#                 'Gm12878', 'H1hesc', 'Hepg2', 'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
#                 'gdi', 'gdi_phred', 'rvis', 'lof_score',
#                 'MMS_delta_logit_psi', 'MMS_ref_acceptorIntron', 'MMS_alt_acceptorIntron',
#                 'MMS_ref_acceptor', 'MMS_alt_acceptor', 'MMS_ref_exon', 'MMS_alt_exon',
#                 'MMS_ref_donor', 'MMS_alt_donor', 'MMS_ref_donorIntron', 'MMS_alt_donorIntron',
#                 'MMS_pathogenicity', 'MMS_efficiency',
#                 'SpliceAI_max', 'CADD_Raw',
#                 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL'
#             ]
#
#             # 筛选存在的列
#             existing_cols = [c for c in cols_to_keep if c in df.columns]
#             df_filtered = df[existing_cols]
#
#             # 长度截断
#             if len(df_filtered) > self.num_samples:
#                 df_filtered = df_filtered.iloc[:self.num_samples]
#
#             data = df_filtered.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
#
#             # 【调试建议】打印一下加载后的形状，确认是否多了7列
#             # print(f"Loaded new features shape: {data.shape}")
#
#             return data
#         except Exception as e:
#             print(f"Error loading new features: {e}")
#             return np.zeros((self.num_samples, 0), dtype=np.float32)
#
#     def _load_tabular_from_list(self, mode):
#         # 【修改】从列表中移除了 'embedding2'
#         feat_names = [
#             #'conservation',
#             'sequence',
#             'score', #'splicing',
#             # 'embedding2',  <-- Removed, replaced by hyena-dna
#             'diffe_feature2mer', 'diffe_featureCKSNAP', 'diffe_featureMismatch', 'diffe_featureNAC',
#             'diffe_featureRC2mer', 'diffe_featureMMI', 'diffe_featureZ_curve_9bit',
#             'diffe_featureZ_curve_12bit', 'diffe_featureZ_curve_36bit', 'diffe_featureZ_curve_48bit',
#             'diffe_featureZ_curve_144bit', 'diffe_featureNMBroto',
#             'mutation_2mer', 'mutation_CKSNAP', 'mutation_Mismatch', 'mutation_NAC', 'mutation_RC2mer',
#             'mutation_MMI', 'mutation_Z_curve_9bit', 'mutation_Z_curve_12bit',
#             'mutation_Z_curve_36bit', 'mutation_Z_curve_48bit', 'mutation_Z_curve_144bit',
#             'mutation_NMBroto',
#             'normal_2mer', 'normal_CKSNAP', 'normal_Mismatch', 'normal_NAC',
#             'normal_RC2mer', 'normal_MMI', 'normal_Z_curve_9bit',
#             'normal_Z_curve_12bit', 'normal_Z_curve_36bit', 'normal_Z_curve_48bit',
#             'normal_Z_curve_144bit', 'normal_NMBroto'
#         ]
#
#         data_list = []
#         for feat in feat_names:
#             filename = f"{feat}_{mode}.txt"
#             file_path = os.path.join(self.root_feat, filename)
#
#             if not os.path.exists(file_path):
#                 continue
#
#             try:
#                 # 原始逻辑保持不变
#                 with open(file_path, 'r') as f:
#                     line = f.readline()
#                     sep = '\t' if '\t' in line else (',' if ',' in line else None)
#
#                 df = pd.read_csv(file_path, sep=sep, header=None, low_memory=False)
#
#                 if len(df) == self.num_samples + 1:
#                     df = df.iloc[1:]
#                 elif len(df) != self.num_samples:
#                     # 长度不对则跳过，防止报错
#                     continue
#
#                 data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
#                 if data.ndim == 1: data = data.reshape(-1, 1)
#                 data_list.append(data)
#
#             except Exception:
#                 pass
#
#         if not data_list:
#             return np.zeros((self.num_samples, 0))
#
#         combined = np.concatenate(data_list, axis=1)
#         return combined
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return (torch.tensor(self.feat_shape[idx]),
#                 torch.tensor(self.feat_bert[idx]),
#                 torch.tensor(self.feat_tabular[idx]),
#                 torch.tensor(self.labels[idx]))


# # Dataset_Integrated_new7-2.py将conservation替换成我自己的
# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import numpy as np
# import os
# from Datasets.load_vcf import load_labels_from_vcf
#
#
# class IntegratedDataset(Dataset):
#     def __init__(self, mode='training'):
#         # ================= 配置路径 =================
#         self.root_feat = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC"
#         # 新特征 Basic-New 的根目录
#         self.root_feat_new = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/basic-new/COSMIC"
#         # 【新增】Hyena-DNA 特征根目录
#         self.root_hyena = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/hyena-dna"
#
#         self.root_bert = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAbert-2"
#         self.root_shape_npy = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR/Datasets/processed_shape/"
#         self.vcf_path = f"/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC/COSMIC_{mode}.vcf"
#
#         print(f"\n>>> 初始化 {mode} 集成数据集 (Model 7-1: Hyena Replace Embedding2) <<<")
#
#         # 1. 加载 Label
#         self.labels = load_labels_from_vcf(self.vcf_path)
#         self.num_samples = len(self.labels)
#
#         # 2. 加载原有的 Tabular 特征 (已移除 embedding2)
#         print("  [Tabular] Loading original legacy features (excluding embedding2)...")
#         feat_original = self._load_tabular_from_list(mode)
#
#         # 3. 加载 Basic-New 补充特征
#         print("  [Tabular] Loading NEW basic features...")
#         feat_new = self._load_new_basic_features(mode)
#
#         # 4. 【新增】加载 Hyena-DNA 特征
#         print("  [Tabular] Loading Hyena-DNA features...")
#         feat_hyena = self._load_hyena_dna(mode)
#
#         # 5. 拼接所有表格类特征
#         # 维度检查与截断：确保行数一致 (以最短的为准)
#         min_len_tab = min(len(feat_original), len(feat_new), len(feat_hyena))
#
#         if not (len(feat_original) == len(feat_new) == len(feat_hyena)):
#             print(f"  Warning: Tabular lengths differ! "
#                   f"Orig: {len(feat_original)}, New: {len(feat_new)}, Hyena: {len(feat_hyena)}. "
#                   f"Truncating to {min_len_tab}.")
#             feat_original = feat_original[:min_len_tab]
#             feat_new = feat_new[:min_len_tab]
#             feat_hyena = feat_hyena[:min_len_tab]
#
#         # 拼接: Original + BasicNew + Hyena
#         self.feat_tabular = np.concatenate([feat_original, feat_new, feat_hyena], axis=1)
#
#         print(f"  [Tabular] Merged Shape: {self.feat_tabular.shape} "
#               f"(Orig: {feat_original.shape[1]} + New: {feat_new.shape[1]} + Hyena: {feat_hyena.shape[1]})")
#
#         # 6. 加载 Shape
#         shape_file = f"shape_all_{mode}.npy"
#         shape_path = os.path.join(self.root_shape_npy, shape_file)
#
#         if os.path.exists(shape_path):
#             self.feat_shape = np.load(shape_path)
#             if len(self.feat_shape) == self.num_samples + 1:
#                 self.feat_shape = self.feat_shape[1:]
#         else:
#             print(f"Warning: Shape file not found at {shape_path}. Using zeros.")
#             self.feat_shape = np.zeros((self.num_samples, 15, 100))
#
#         # 7. 加载 DNABERT-2
#         if mode == 'training':
#             bert_file = "train_embeddings.npy"
#         else:
#             bert_file = "test_embeddings.npy"
#
#         self.feat_bert = np.load(os.path.join(self.root_bert, bert_file))
#         if self.feat_bert.ndim == 3 and self.feat_bert.shape[1] != 768:
#             self.feat_bert = self.feat_bert.transpose(0, 2, 1)
#         if len(self.feat_bert) == self.num_samples + 1:
#             self.feat_bert = self.feat_bert[1:]
#
#         # 8. 最终一致性检查
#         min_len = min(self.num_samples, len(self.feat_tabular), len(self.feat_shape), len(self.feat_bert))
#         if self.num_samples != min_len:
#             print(f"  最终长度对齐: {self.num_samples} -> {min_len}")
#             self.labels = self.labels[:min_len]
#             self.feat_tabular = self.feat_tabular[:min_len]
#             self.feat_shape = self.feat_shape[:min_len]
#             self.feat_bert = self.feat_bert[:min_len]
#
#         print("   数据加载与对齐完成！")
#
#     def _load_hyena_dna(self, mode):
#         """
#         加载 Hyena-DNA 特征代替 embedding2
#         文件格式: Header存在, Tab分隔, 256维
#         """
#         filename = f"hyena-dna_{mode}.txt"
#         file_path = os.path.join(self.root_hyena, filename)
#
#         if not os.path.exists(file_path):
#             print(f"Error: Hyena file not found at {file_path}")
#             return np.zeros((self.num_samples, 256), dtype=np.float32)
#
#         try:
#             # 读取数据，假设第一行是 Header (hyena-dna_0 ... hyena-dna_255)
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#
#             # 简单校验列数，如果是256维
#             if df.shape[1] < 256:
#                 print(f"Warning: Hyena feature dimension {df.shape[1]} < 256?")
#
#             # 长度处理：通常如果带 Header，read_csv 读取后的长度应正好等于样本数
#             # 如果文件末尾有空行或者多余行，进行截断
#             if len(df) > self.num_samples:
#                 df = df.iloc[:self.num_samples]
#
#             data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
#             return data
#
#         except Exception as e:
#             print(f"Error loading Hyena features: {e}")
#             return np.zeros((self.num_samples, 256), dtype=np.float32)
#
#     def _load_new_basic_features(self, mode):
#         """加载新的 Basic 特征 (包含由你自己生成的 Conservation 特征)"""
#         if mode == 'training':
#             filename = "train_BASIC-1.txt"
#         else:
#             filename = "test_BASIC-1.txt"
#
#         file_path = os.path.join(self.root_feat_new, filename)
#         if not os.path.exists(file_path):
#             # 注意：如果文件不存在，返回空会导致后续维度对齐出问题，但在你的环境中文件是存在的
#             return np.zeros((self.num_samples, 0), dtype=np.float32)
#
#         try:
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#
#             cols_to_keep = [
#                 # --- 【新增】你自己生成的 Conservation 特征 (7个) ---
#                 'verPhCons', 'verPhyloP', 'mamPhCons', 'mamPhyloP',
#                 'priPhCons', 'priPhyloP', 'GerpS',
#
#                 # --- 原有的 Basic 特征 ---
#                 'Gm12878', 'H1hesc', 'Hepg2', 'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
#                 'gdi', 'gdi_phred', 'rvis', 'lof_score',
#                 'MMS_delta_logit_psi', 'MMS_ref_acceptorIntron', 'MMS_alt_acceptorIntron',
#                 'MMS_ref_acceptor', 'MMS_alt_acceptor', 'MMS_ref_exon', 'MMS_alt_exon',
#                 'MMS_ref_donor', 'MMS_alt_donor', 'MMS_ref_donorIntron', 'MMS_alt_donorIntron',
#                 'MMS_pathogenicity', 'MMS_efficiency',
#                 'SpliceAI_max', 'CADD_Raw',
#                 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL'
#             ]
#
#             # 筛选存在的列
#             existing_cols = [c for c in cols_to_keep if c in df.columns]
#             df_filtered = df[existing_cols]
#
#             # 长度截断
#             if len(df_filtered) > self.num_samples:
#                 df_filtered = df_filtered.iloc[:self.num_samples]
#
#             data = df_filtered.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
#
#             # 【调试建议】打印一下加载后的形状，确认是否多了7列
#             # print(f"Loaded new features shape: {data.shape}")
#
#             return data
#         except Exception as e:
#             print(f"Error loading new features: {e}")
#             return np.zeros((self.num_samples, 0), dtype=np.float32)
#
#     def _load_tabular_from_list(self, mode):
#         # 【修改】从列表中移除了 'embedding2'
#         feat_names = [
#             #'conservation',
#             'sequence', 'score', 'splicing',
#             # 'embedding2',  <-- Removed, replaced by hyena-dna
#             'diffe_feature2mer', 'diffe_featureCKSNAP', 'diffe_featureMismatch', 'diffe_featureNAC',
#             'diffe_featureRC2mer', 'diffe_featureMMI', 'diffe_featureZ_curve_9bit',
#             'diffe_featureZ_curve_12bit', 'diffe_featureZ_curve_36bit', 'diffe_featureZ_curve_48bit',
#             'diffe_featureZ_curve_144bit', 'diffe_featureNMBroto',
#             'mutation_2mer', 'mutation_CKSNAP', 'mutation_Mismatch', 'mutation_NAC', 'mutation_RC2mer',
#             'mutation_MMI', 'mutation_Z_curve_9bit', 'mutation_Z_curve_12bit',
#             'mutation_Z_curve_36bit', 'mutation_Z_curve_48bit', 'mutation_Z_curve_144bit',
#             'mutation_NMBroto',
#             'normal_2mer', 'normal_CKSNAP', 'normal_Mismatch', 'normal_NAC',
#             'normal_RC2mer', 'normal_MMI', 'normal_Z_curve_9bit',
#             'normal_Z_curve_12bit', 'normal_Z_curve_36bit', 'normal_Z_curve_48bit',
#             'normal_Z_curve_144bit', 'normal_NMBroto'
#         ]
#
#         data_list = []
#         for feat in feat_names:
#             filename = f"{feat}_{mode}.txt"
#             file_path = os.path.join(self.root_feat, filename)
#
#             if not os.path.exists(file_path):
#                 continue
#
#             try:
#                 # 原始逻辑保持不变
#                 with open(file_path, 'r') as f:
#                     line = f.readline()
#                     sep = '\t' if '\t' in line else (',' if ',' in line else None)
#
#                 df = pd.read_csv(file_path, sep=sep, header=None, low_memory=False)
#
#                 if len(df) == self.num_samples + 1:
#                     df = df.iloc[1:]
#                 elif len(df) != self.num_samples:
#                     # 长度不对则跳过，防止报错
#                     continue
#
#                 data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
#                 if data.ndim == 1: data = data.reshape(-1, 1)
#                 data_list.append(data)
#
#             except Exception:
#                 pass
#
#         if not data_list:
#             return np.zeros((self.num_samples, 0))
#
#         combined = np.concatenate(data_list, axis=1)
#         return combined
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return (torch.tensor(self.feat_shape[idx]),
#                 torch.tensor(self.feat_bert[idx]),
#                 torch.tensor(self.feat_tabular[idx]),
#                 torch.tensor(self.labels[idx]))



# # Dataset_Integrated_new7-1.py
# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import numpy as np
# import os
# from Datasets.load_vcf import load_labels_from_vcf
#
#
# class IntegratedDataset(Dataset):
#     def __init__(self, mode='training'):
#         # ================= 配置路径 =================
#         self.root_feat = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC"
#         # 新特征 Basic-New 的根目录
#         self.root_feat_new = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/basic-new/COSMIC"
#         # 【新增】Hyena-DNA 特征根目录
#         self.root_hyena = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/hyena-dna"
#
#         self.root_bert = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAbert-2"
#         self.root_shape_npy = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR/Datasets/processed_shape/"
#         self.vcf_path = f"/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC/COSMIC_{mode}.vcf"
#
#         print(f"\n>>> 初始化 {mode} 集成数据集 (Model 7-1: Hyena Replace Embedding2) <<<")
#
#         # 1. 加载 Label
#         self.labels = load_labels_from_vcf(self.vcf_path)
#         self.num_samples = len(self.labels)
#
#         # 2. 加载原有的 Tabular 特征 (已移除 embedding2)
#         print("  [Tabular] Loading original legacy features (excluding embedding2)...")
#         feat_original = self._load_tabular_from_list(mode)
#
#         # 3. 加载 Basic-New 补充特征
#         print("  [Tabular] Loading NEW basic features...")
#         feat_new = self._load_new_basic_features(mode)
#
#         # 4. 【新增】加载 Hyena-DNA 特征
#         print("  [Tabular] Loading Hyena-DNA features...")
#         feat_hyena = self._load_hyena_dna(mode)
#
#         # 5. 拼接所有表格类特征
#         # 维度检查与截断：确保行数一致 (以最短的为准)
#         min_len_tab = min(len(feat_original), len(feat_new), len(feat_hyena))
#
#         if not (len(feat_original) == len(feat_new) == len(feat_hyena)):
#             print(f"  Warning: Tabular lengths differ! "
#                   f"Orig: {len(feat_original)}, New: {len(feat_new)}, Hyena: {len(feat_hyena)}. "
#                   f"Truncating to {min_len_tab}.")
#             feat_original = feat_original[:min_len_tab]
#             feat_new = feat_new[:min_len_tab]
#             feat_hyena = feat_hyena[:min_len_tab]
#
#         # 拼接: Original + BasicNew + Hyena
#         self.feat_tabular = np.concatenate([feat_original, feat_new, feat_hyena], axis=1)
#
#         print(f"  [Tabular] Merged Shape: {self.feat_tabular.shape} "
#               f"(Orig: {feat_original.shape[1]} + New: {feat_new.shape[1]} + Hyena: {feat_hyena.shape[1]})")
#
#         # 6. 加载 Shape
#         shape_file = f"shape_all_{mode}.npy"
#         shape_path = os.path.join(self.root_shape_npy, shape_file)
#
#         if os.path.exists(shape_path):
#             self.feat_shape = np.load(shape_path)
#             if len(self.feat_shape) == self.num_samples + 1:
#                 self.feat_shape = self.feat_shape[1:]
#         else:
#             print(f"Warning: Shape file not found at {shape_path}. Using zeros.")
#             self.feat_shape = np.zeros((self.num_samples, 15, 100))
#
#         # 7. 加载 DNABERT-2
#         if mode == 'training':
#             bert_file = "train_embeddings.npy"
#         else:
#             bert_file = "test_embeddings.npy"
#
#         self.feat_bert = np.load(os.path.join(self.root_bert, bert_file))
#         if self.feat_bert.ndim == 3 and self.feat_bert.shape[1] != 768:
#             self.feat_bert = self.feat_bert.transpose(0, 2, 1)
#         if len(self.feat_bert) == self.num_samples + 1:
#             self.feat_bert = self.feat_bert[1:]
#
#         # 8. 最终一致性检查
#         min_len = min(self.num_samples, len(self.feat_tabular), len(self.feat_shape), len(self.feat_bert))
#         if self.num_samples != min_len:
#             print(f"  最终长度对齐: {self.num_samples} -> {min_len}")
#             self.labels = self.labels[:min_len]
#             self.feat_tabular = self.feat_tabular[:min_len]
#             self.feat_shape = self.feat_shape[:min_len]
#             self.feat_bert = self.feat_bert[:min_len]
#
#         print("   数据加载与对齐完成！")
#
#     def _load_hyena_dna(self, mode):
#         """
#         加载 Hyena-DNA 特征代替 embedding2
#         文件格式: Header存在, Tab分隔, 256维
#         """
#         filename = f"hyena-dna_{mode}.txt"
#         file_path = os.path.join(self.root_hyena, filename)
#
#         if not os.path.exists(file_path):
#             print(f"Error: Hyena file not found at {file_path}")
#             return np.zeros((self.num_samples, 256), dtype=np.float32)
#
#         try:
#             # 读取数据，假设第一行是 Header (hyena-dna_0 ... hyena-dna_255)
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#
#             # 简单校验列数，如果是256维
#             if df.shape[1] < 256:
#                 print(f"Warning: Hyena feature dimension {df.shape[1]} < 256?")
#
#             # 长度处理：通常如果带 Header，read_csv 读取后的长度应正好等于样本数
#             # 如果文件末尾有空行或者多余行，进行截断
#             if len(df) > self.num_samples:
#                 df = df.iloc[:self.num_samples]
#
#             data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
#             return data
#
#         except Exception as e:
#             print(f"Error loading Hyena features: {e}")
#             return np.zeros((self.num_samples, 256), dtype=np.float32)
#
#     def _load_new_basic_features(self, mode):
#         """加载新的 Basic 特征"""
#         if mode == 'training':
#             filename = "train_BASIC-1.txt"
#         else:
#             filename = "test_BASIC-1.txt"
#
#         file_path = os.path.join(self.root_feat_new, filename)
#         if not os.path.exists(file_path):
#             return np.zeros((self.num_samples, 0), dtype=np.float32)
#
#         try:
#             df = pd.read_csv(file_path, sep='\t', low_memory=False)
#
#             cols_to_keep = [
#                 'Gm12878', 'H1hesc', 'Hepg2', 'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
#                 'gdi', 'gdi_phred', 'rvis', 'lof_score',
#                 'MMS_delta_logit_psi', 'MMS_ref_acceptorIntron', 'MMS_alt_acceptorIntron',
#                 'MMS_ref_acceptor', 'MMS_alt_acceptor', 'MMS_ref_exon', 'MMS_alt_exon',
#                 'MMS_ref_donor', 'MMS_alt_donor', 'MMS_ref_donorIntron', 'MMS_alt_donorIntron',
#                 'MMS_pathogenicity', 'MMS_efficiency',
#                 'SpliceAI_max', 'CADD_Raw',
#                 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL'
#             ]
#             existing_cols = [c for c in cols_to_keep if c in df.columns]
#             df_filtered = df[existing_cols]
#
#             if len(df_filtered) > self.num_samples:
#                 df_filtered = df_filtered.iloc[:self.num_samples]
#
#             data = df_filtered.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
#             return data
#         except Exception as e:
#             print(f"Error loading new features: {e}")
#             return np.zeros((self.num_samples, 0), dtype=np.float32)
#
#     def _load_tabular_from_list(self, mode):
#         # 【修改】从列表中移除了 'embedding2'
#         feat_names = [
#             'conservation', 'sequence', 'score', 'splicing',
#             # 'embedding2',  <-- Removed, replaced by hyena-dna
#             'diffe_feature2mer', 'diffe_featureCKSNAP', 'diffe_featureMismatch', 'diffe_featureNAC',
#             'diffe_featureRC2mer', 'diffe_featureMMI', 'diffe_featureZ_curve_9bit',
#             'diffe_featureZ_curve_12bit', 'diffe_featureZ_curve_36bit', 'diffe_featureZ_curve_48bit',
#             'diffe_featureZ_curve_144bit', 'diffe_featureNMBroto',
#             'mutation_2mer', 'mutation_CKSNAP', 'mutation_Mismatch', 'mutation_NAC', 'mutation_RC2mer',
#             'mutation_MMI', 'mutation_Z_curve_9bit', 'mutation_Z_curve_12bit',
#             'mutation_Z_curve_36bit', 'mutation_Z_curve_48bit', 'mutation_Z_curve_144bit',
#             'mutation_NMBroto',
#             'normal_2mer', 'normal_CKSNAP', 'normal_Mismatch', 'normal_NAC',
#             'normal_RC2mer', 'normal_MMI', 'normal_Z_curve_9bit',
#             'normal_Z_curve_12bit', 'normal_Z_curve_36bit', 'normal_Z_curve_48bit',
#             'normal_Z_curve_144bit', 'normal_NMBroto'
#         ]
#
#         data_list = []
#         for feat in feat_names:
#             filename = f"{feat}_{mode}.txt"
#             file_path = os.path.join(self.root_feat, filename)
#
#             if not os.path.exists(file_path):
#                 continue
#
#             try:
#                 # 原始逻辑保持不变
#                 with open(file_path, 'r') as f:
#                     line = f.readline()
#                     sep = '\t' if '\t' in line else (',' if ',' in line else None)
#
#                 df = pd.read_csv(file_path, sep=sep, header=None, low_memory=False)
#
#                 if len(df) == self.num_samples + 1:
#                     df = df.iloc[1:]
#                 elif len(df) != self.num_samples:
#                     # 长度不对则跳过，防止报错
#                     continue
#
#                 data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
#                 if data.ndim == 1: data = data.reshape(-1, 1)
#                 data_list.append(data)
#
#             except Exception:
#                 pass
#
#         if not data_list:
#             return np.zeros((self.num_samples, 0))
#
#         combined = np.concatenate(data_list, axis=1)
#         return combined
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return (torch.tensor(self.feat_shape[idx]),
#                 torch.tensor(self.feat_bert[idx]),
#                 torch.tensor(self.feat_tabular[idx]),
#                 torch.tensor(self.labels[idx]))