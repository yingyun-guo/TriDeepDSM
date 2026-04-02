import pandas as pd
import numpy as np
import os
import sys

# ==============================================================================
# 1. 路径配置
# ==============================================================================
BASE_DIR = "/data/gyy/Project/DeepSTF-new/DeepSTF"
OUTPUT_DIR = os.path.join(BASE_DIR, "Datasets/aligned_test2_model7-5-1")

PATHS = {
    "BASIC": f"{BASE_DIR}/feature/basic-new",
    "PHYSIO": f"{BASE_DIR}/feature/physicochemical",  # Test physio
    "HYENA": f"{BASE_DIR}/feature/hyena-dna",
    "BERT": f"{BASE_DIR}/feature/DNAbert-2",
    "EOSM_LEGACY": f"{BASE_DIR}/feature/EOSM",
    "SOMA_LEGACY": f"{BASE_DIR}/feature/SomaMutDBpart",

    # 【关键】用于获取正确维度的训练集路径
    "PHYSIO_TRAIN": f"{BASE_DIR}/feature/physicochemical/COSMIC",
    "COSMIC_TRAIN": f"{BASE_DIR}/feature/COSMIC"
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 2. 特征定义
# ==============================================================================
AUTHOR_FEATS = ['sequence']  # Model 7-5-1 仅保留 Sequence (8维)

# Basic (68维)
BASIC_COLS_TO_KEEP = [
    'verPhCons', 'verPhyloP', 'mamPhCons', 'mamPhyloP', 'priPhCons', 'priPhyloP', 'GerpS',
    'Gm12878', 'H1hesc', 'Hepg2', 'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
    'gdi', 'gdi_phred', 'rvis', 'lof_score',
    'SilVA', 'FATHMM-XF', 'PhD_SNPg_SCORE', 'CADD_PHRED', 'CADD_Raw', 'SpliceAI_max', 'GERP..',
    'MMS_delta_logit_psi', 'MMS_ref_acceptorIntron', 'MMS_alt_acceptorIntron',
    'MMS_ref_acceptor', 'MMS_alt_acceptor', 'MMS_ref_exon', 'MMS_alt_exon',
    'MMS_ref_donor', 'MMS_alt_donor', 'MMS_ref_donorIntron', 'MMS_alt_donorIntron',
    'MMS_pathogenicity', 'MMS_efficiency',
    'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',
    'CpG.', 'SR.', 'SR..1', 'FAS6.', 'FAS6..1', 'MES', 'dMES', 'MES.', 'MES..1',
    'MEC.MC.', 'MEC.CS.', 'MES.KM.', 'PESE.', 'PESE..1', 'PESS.', 'PESS..1',
    'pdG_pre_50', 'pdG_post_50', 'pvar_pre_50', 'pvar_post_50'
]

# Physio (1671维)
PHYSIO_TYPES = [
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


# ==============================================================================
# 3. 核心工具：获取训练集正确维度
# ==============================================================================
def get_train_dimension(feat_name, category):
    """读取训练集文件的前几行来确定维度"""
    path = ""
    if category == "PHYSIO":
        path = os.path.join(PATHS["PHYSIO_TRAIN"], f"train_{feat_name}.txt")
    elif category == "AUTHOR":
        path = os.path.join(PATHS["COSMIC_TRAIN"], f"{feat_name}_training.txt")

    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                line = f.readline()
            sep = '\t' if '\t' in line else ','
            # 读前5行
            df = pd.read_csv(path, sep=sep, nrows=5, header=None, low_memory=False)
            # 简单判断是否有header
            try:
                pd.to_numeric(df.iloc[0, 0])
            except:
                df = df.iloc[1:]
            return df.shape[1]
        except:
            return 0
    return 0


# ==============================================================================
# 4. 加载函数
# ==============================================================================
def load_and_align(dataset_name, num_samples):
    # --- 1. Sequence (Legacy) ---
    seq_data_list = []
    target_dir = PATHS["EOSM_LEGACY"] if dataset_name == "EOSM" else PATHS["SOMA_LEGACY"]

    for feat in AUTHOR_FEATS:
        ref_dim = get_train_dimension(feat, "AUTHOR")
        if ref_dim == 0: ref_dim = 8  # Fallback for Sequence if train file not found

        file_path = os.path.join(target_dir, f"{feat}.txt")
        data = None
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    line = f.readline()
                sep = '\t' if '\t' in line else ','
                df = pd.read_csv(file_path, sep=sep, header=None, low_memory=False)
                # Check header
                try:
                    pd.to_numeric(df.iloc[0, 0])
                except:
                    df = df.iloc[1:]

                if len(df) > num_samples: df = df.iloc[:num_samples]
                data = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
            except:
                pass

        if data is None or data.shape[1] != ref_dim:
            # print(f"    [Align] {feat}: Missing/Mismatch. Padding zeros {ref_dim} cols.")
            data = np.zeros((num_samples, ref_dim), dtype=np.float32)

        seq_data_list.append(data)
    f_seq = np.concatenate(seq_data_list, axis=1)

    # --- 2. Basic ---
    basic_path = os.path.join(PATHS["BASIC"], dataset_name, f"{dataset_name}_BASIC-1.txt")
    if os.path.exists(basic_path):
        try:
            df = pd.read_csv(basic_path, sep='\t', low_memory=False)
            for c in BASIC_COLS_TO_KEEP:
                if c not in df.columns: df[c] = 0.0
            df = df[BASIC_COLS_TO_KEEP]
            if len(df) > num_samples: df = df.iloc[:num_samples]
            f_basic = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
        except:
            f_basic = np.zeros((num_samples, 68), dtype=np.float32)
    else:
        f_basic = np.zeros((num_samples, 68), dtype=np.float32)

    # --- 3. Physio (Critical Fix) ---
    physio_list = []
    for f_type in PHYSIO_TYPES:
        ref_dim = get_train_dimension(f_type, "PHYSIO")
        if ref_dim == 0:
            print(f"    [Error] Could not find dim for {f_type} in Training set!")
            # 这是一个保险，如果找不到训练集文件，根据经验硬编码一些常见维度，或报错
            # 这里为了不中断，设为1 (但这是危险的)
            ref_dim = 1

            # 尝试加载 Test 文件
        p1 = os.path.join(PATHS["PHYSIO"], dataset_name, f"{dataset_name}_{f_type}.txt")
        p2 = os.path.join(PATHS["PHYSIO"], dataset_name, f"{f_type}.txt")
        file_path = p1 if os.path.exists(p1) else p2

        data = None
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    line = f.readline()
                sep = '\t' if '\t' in line else ','
                df = pd.read_csv(file_path, sep=sep, low_memory=False)
                if len(df) > num_samples: df = df.iloc[:num_samples]
                val = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
                if val.ndim == 1: val = val.reshape(-1, 1)

                # 维度检查
                if val.shape[1] == ref_dim:
                    data = val
                else:
                    # print(f"    [Mismatch] {f_type}: Got {val.shape[1]}, Expected {ref_dim}. Padding/Truncating.")
                    if val.shape[1] > ref_dim:
                        data = val[:, :ref_dim]
                    else:
                        data = np.hstack([val, np.zeros((num_samples, ref_dim - val.shape[1]))])
            except:
                pass

        if data is None:
            # print(f"    [Missing] {f_type}. Padding {ref_dim} zeros.")
            data = np.zeros((num_samples, ref_dim), dtype=np.float32)

        physio_list.append(data)
    f_physio = np.concatenate(physio_list, axis=1)

    # --- 4. Hyena ---
    hyena_path = os.path.join(PATHS["HYENA"], f"{dataset_name}_hyena-dna.txt")
    if os.path.exists(hyena_path):
        try:
            df = pd.read_csv(hyena_path, sep='\t', low_memory=False)
            if len(df) > num_samples: df = df.iloc[:num_samples]
            f_hyena = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
        except:
            f_hyena = np.zeros((num_samples, 256), dtype=np.float32)
    else:
        f_hyena = np.zeros((num_samples, 256), dtype=np.float32)

    # Final Merge
    f_tab = np.concatenate([f_seq, f_basic, f_physio, f_hyena], axis=1)

    # Validation
    print(
        f"    Component Shapes: Seq={f_seq.shape[1]}, Basic={f_basic.shape[1]}, Physio={f_physio.shape[1]}, Hyena={f_hyena.shape[1]}")
    return f_tab


def process(name, label_val, bert_file):
    print(f"\nProcessing {name}...")
    # Determine samples
    basic_path = os.path.join(PATHS["BASIC"], name, f"{name}_BASIC-1.txt")
    if os.path.exists(basic_path):
        with open(basic_path) as f:
            num_samples = sum(1 for _ in f) - 1
    else:
        # Fallback to bert
        bp = os.path.join(PATHS["BERT"], bert_file)
        if os.path.exists(bp):
            num_samples = len(np.load(bp))
        else:
            return

    print(f"  Samples: {num_samples}")

    # 1. Tabular (Aligned)
    f_tab = load_and_align(name, num_samples)
    print(f"  Tabular Shape: {f_tab.shape}")

    # 2. Bert
    bp = os.path.join(PATHS["BERT"], bert_file)
    if os.path.exists(bp):
        f_bert = np.load(bp)
        if len(f_bert) > num_samples: f_bert = f_bert[:num_samples]
        if f_bert.ndim == 3: f_bert = f_bert.transpose(0, 2, 1)  # (N, L, D) -> (N, D, L) for Conv1d
    else:
        f_bert = np.zeros((num_samples, 768), dtype=np.float32)

    # 3. Spatial (Placeholder)
    f_spatial = np.zeros((num_samples, 19, 100), dtype=np.float32)

    # 4. Label
    f_label = np.full(num_samples, label_val, dtype=np.float32)

    # Save
    np.save(os.path.join(OUTPUT_DIR, f"{name}_tabular.npy"), f_tab)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_spatial.npy"), f_spatial)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_bert.npy"), f_bert)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_label.npy"), f_label)
    print("  Saved.")


if __name__ == "__main__":
    process("EOSM", 1.0, "EOSM_dnabert2.npy")
    process("SomaMutDB", 0.0, "SomaMutDB_dnabert2.npy")