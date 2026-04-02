import pandas as pd
import numpy as np
import os
import pysam
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 参考基因组路径
REF_GENOME_PATH = "/data/gyy/Data/hg19.fa/hg19.fa"

# 2. 输入数据列表 (修改为新的 VCF 数据集)
INPUT_FILES = {
    "KRAS": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/model_input/KRAS_hg19_final.vcf",
    "LUSC": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/model_input/LUSC_hg19_final.vcf",
    "TP53": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/model_input/TP53_hg19_final.vcf"
}

# 3. 输出目录 (统一放到 case_analysis 目录下)
SAVE_DIR = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/one-hot/"

# 4. 序列窗口大小
SEQUENCE_LENGTH = 101
FLANK_LEFT = 50  # 左侧取 50
FLANK_RIGHT = 50  # 右侧取 50

# ==========================================================

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)


def get_onehot_map():
    """定义 One-Hot 映射"""
    return {
        'A': np.array([1, 0, 0, 0], dtype=np.float32),
        'C': np.array([0, 1, 0, 0], dtype=np.float32),
        'G': np.array([0, 0, 1, 0], dtype=np.float32),
        'T': np.array([0, 0, 0, 1], dtype=np.float32),
        'N': np.array([0, 0, 0, 0], dtype=np.float32),
        'a': np.array([1, 0, 0, 0], dtype=np.float32),
        'c': np.array([0, 1, 0, 0], dtype=np.float32),
        'g': np.array([0, 0, 1, 0], dtype=np.float32),
        't': np.array([0, 0, 0, 1], dtype=np.float32)
    }


def generate_onehot(ref_fasta, txt_path, mode_name):
    print(f"\n>>> Processing {mode_name} ...")

    if not os.path.exists(ref_fasta):
        print(f"Error: Reference Genome not found at: {ref_fasta}")
        return

    fasta = pysam.FastaFile(ref_fasta)
    try:
        # 使用 \s+ 兼容 VCF 文件的多空格/制表符格式，并忽略注释行
        df = pd.read_csv(txt_path, sep=r'\s+', comment='#')
    except Exception as e:
        print(f"  Error reading file {txt_path}: {e}")
        return

    onehot_map = get_onehot_map()
    encoded_list = []

    # 规范化列名 (去除空格，转小写)
    df.columns = [c.strip().lower() for c in df.columns]

    if 'chr' not in df.columns or 'pos' not in df.columns:
        print(f"  Error: 'chr' or 'pos' column missing in {txt_path}")
        return

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Encoding {mode_name}"):
        chrom = str(row['chr'])
        pos = int(row['pos'])  # 1-based position

        if not chrom.startswith('chr'):
            chrom_query = f"chr{chrom}"
        else:
            chrom_query = chrom

        if chrom_query == "chrMT": chrom_query = "chrM"

        if chrom_query not in fasta.references:
            if chrom.startswith('chr') and chrom[3:] in fasta.references:
                chrom_query = chrom[3:]
            elif chrom_query not in fasta.references:
                encoded_list.append(np.zeros((4, SEQUENCE_LENGTH), dtype=np.float32))
                continue

        mut_idx_0based = pos - 1
        start = mut_idx_0based - FLANK_LEFT
        end = mut_idx_0based + 1 + FLANK_RIGHT

        try:
            seq_str = fasta.fetch(chrom_query, start, end).upper()
        except KeyError:
            seq_str = "N" * SEQUENCE_LENGTH

        if len(seq_str) != SEQUENCE_LENGTH:
            seq_str = seq_str.ljust(SEQUENCE_LENGTH, 'N')

        seq_str = seq_str[:SEQUENCE_LENGTH]

        # One-Hot 编码
        mat = []
        for base in seq_str:
            mat.append(onehot_map.get(base, onehot_map['N']))

        mat = np.array(mat, dtype=np.float32)  # (101, 4)

        # 转换为 PyTorch 友好的 (Channels, Seq_Len) -> (4, 101)
        mat = mat.transpose(1, 0)

        encoded_list.append(mat)

    # 堆叠并保存
    if len(encoded_list) > 0:
        final_array = np.stack(encoded_list, axis=0)

        # 保存文件名: KRAS_onehot.npy
        save_name = f"{mode_name}_onehot.npy"
        save_path = os.path.join(SAVE_DIR, save_name)
        np.save(save_path, final_array)

        print(f" Saved {mode_name}: {final_array.shape} -> {save_path}")
    else:
        print(f"   No valid sequences extracted for {mode_name}")


# ==========================================================
if __name__ == "__main__":
    print(f"Target Directory: {SAVE_DIR}")
    print(f"Reference Genome: {REF_GENOME_PATH}")

    if not os.path.exists(REF_GENOME_PATH):
        print("错误：参考基因组路径不存在，请检查！")
    else:
        for mode, txt_path in INPUT_FILES.items():
            if os.path.exists(txt_path):
                generate_onehot(REF_GENOME_PATH, txt_path, mode)
            else:
                print(f"Skipping {mode} (File not found: {txt_path})")



# import pandas as pd
# import numpy as np
# import os
# import pysam
# from tqdm import tqdm
#
# # ================= 配置区域 =================
# # 1. 参考基因组路径 (之前确认过是这个)
# REF_GENOME_PATH = "/data/gyy/Data/hg19.fa/hg19.fa"
#
# # 2. 输入数据列表 (修改为你现在的三个 txt 数据)
# INPUT_FILES = {
#     "MFDSMC_train": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/MFDSMC_train.txt",
#     "EPEL_train": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/EPEL_train.txt"
# }
# # INPUT_FILES = {
# #     "MFDSMC_test1": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/MFDSMC_test1.txt",
# #     "MFDSMC_test2": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/MFDSMC_test2.txt",
# #     "EPEL_test": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/EPEL_test.txt"
# # }
#
# # 3. 输出目录
# SAVE_DIR = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/one-hot/"
#
# # 4. 序列窗口大小
# # 典型的 CNN 输入通常是奇数以保持对称，这里设为 101 (50 + 1 + 50)
# SEQUENCE_LENGTH = 101
# FLANK_LEFT = 50  # 左侧取 50
# FLANK_RIGHT = 50  # 右侧取 50
# # 注意：FLANK_LEFT + 1 + FLANK_RIGHT 必须等于 SEQUENCE_LENGTH
#
# # ==========================================================
#
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR, exist_ok=True)
#
#
# def get_onehot_map():
#     """定义 One-Hot 映射"""
#     # shape: (4,)
#     # 这里定义标准 A, C, G, T。N 和其他字符映射为全 0
#     return {
#         'A': np.array([1, 0, 0, 0], dtype=np.float32),
#         'C': np.array([0, 1, 0, 0], dtype=np.float32),
#         'G': np.array([0, 0, 1, 0], dtype=np.float32),
#         'T': np.array([0, 0, 0, 1], dtype=np.float32),
#         'N': np.array([0, 0, 0, 0], dtype=np.float32),
#         'a': np.array([1, 0, 0, 0], dtype=np.float32),
#         'c': np.array([0, 1, 0, 0], dtype=np.float32),
#         'g': np.array([0, 0, 1, 0], dtype=np.float32),
#         't': np.array([0, 0, 0, 1], dtype=np.float32)
#     }
#
#
# def generate_onehot(ref_fasta, txt_path, mode_name):
#     print(f"\n>>> Processing {mode_name} ...")
#
#     if not os.path.exists(ref_fasta):
#         print(f"Error: Reference Genome not found at: {ref_fasta}")
#         return
#
#     # 1. 打开 Fasta 和 TXT
#     fasta = pysam.FastaFile(ref_fasta)
#     try:
#         # 你的数据是 tab 分隔的 txt
#         df = pd.read_csv(txt_path, sep='\t')
#     except Exception as e:
#         print(f"  Error reading file {txt_path}: {e}")
#         return
#
#     onehot_map = get_onehot_map()
#     encoded_list = []
#
#     # 2. 规范化列名 (去除空格，转小写)
#     df.columns = [c.strip().lower() for c in df.columns]
#
#     # 检查必要的列
#     if 'chr' not in df.columns or 'pos' not in df.columns:
#         print(f"  Error: 'chr' or 'pos' column missing in {txt_path}")
#         return
#
#     # 3. 遍历提取
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Encoding {mode_name}"):
#         chrom = str(row['chr'])
#         pos = int(row['pos'])  # 1-based position
#
#         # 兼容 chr 前缀处理
#         if not chrom.startswith('chr'):
#             chrom_query = f"chr{chrom}"
#         else:
#             chrom_query = chrom
#
#         # 特殊染色体处理
#         if chrom_query == "chrMT": chrom_query = "chrM"
#
#         # 检查染色体是否存在
#         if chrom_query not in fasta.references:
#             # 尝试不带 chr
#             if chrom.startswith('chr') and chrom[3:] in fasta.references:
#                 chrom_query = chrom[3:]
#             elif chrom_query not in fasta.references:
#                 # 找不到染色体，填全0矩阵
#                 encoded_list.append(np.zeros((4, SEQUENCE_LENGTH), dtype=np.float32))
#                 continue
#
#         # 计算提取范围 (0-based)
#         # pos 是突变点 (1-based)，对应的索引是 pos-1
#         mut_idx_0based = pos - 1
#         start = mut_idx_0based - FLANK_LEFT
#         end = mut_idx_0based + 1 + FLANK_RIGHT
#
#         try:
#             # 提取序列
#             seq_str = fasta.fetch(chrom_query, start, end).upper()
#         except KeyError:
#             seq_str = "N" * SEQUENCE_LENGTH
#
#         # 长度校验与 Padding
#         if len(seq_str) != SEQUENCE_LENGTH:
#             # 如果提取出来的长度不够（比如在染色体开头或结尾）
#             seq_str = seq_str.ljust(SEQUENCE_LENGTH, 'N')
#
#         # 确保截断到正确长度
#         seq_str = seq_str[:SEQUENCE_LENGTH]
#
#         # 4. One-Hot 编码
#         # 原始 shape: (Seq_Len, 4)
#         mat = []
#         for base in seq_str:
#             mat.append(onehot_map.get(base, onehot_map['N']))
#
#         mat = np.array(mat, dtype=np.float32)  # (101, 4)
#
#         # 转换为 PyTorch 友好的 (Channels, Seq_Len) -> (4, 101)
#         # 如果你需要 (101, 4)，请注释掉下面这一行
#         mat = mat.transpose(1, 0)
#
#         encoded_list.append(mat)
#
#     # 5. 堆叠并保存
#     # Shape: (N, 4, 101)
#     if len(encoded_list) > 0:
#         final_array = np.stack(encoded_list, axis=0)
#
#         # 保存文件名: MFDSMC_test1_onehot.npy
#         save_name = f"{mode_name}_onehot.npy"
#         save_path = os.path.join(SAVE_DIR, save_name)
#         np.save(save_path, final_array)
#
#         print(f" Saved {mode_name}: {final_array.shape} -> {save_path}")
#     else:
#         print(f"   No valid sequences extracted for {mode_name}")
#
#
# # ==========================================================
# if __name__ == "__main__":
#     print(f"Target Directory: {SAVE_DIR}")
#     print(f"Reference Genome: {REF_GENOME_PATH}")
#
#     if not os.path.exists(REF_GENOME_PATH):
#         print("错误：参考基因组路径不存在，请检查！")
#     else:
#         for mode, txt_path in INPUT_FILES.items():
#             if os.path.exists(txt_path):
#                 generate_onehot(REF_GENOME_PATH, txt_path, mode)
#             else:
#                 print(f"Skipping {mode} (File not found: {txt_path})")


# import pandas as pd
# import numpy as np
# import os
# import pysam
# from tqdm import tqdm
#
# # ================= 配置区域 =================
# # 1. 你的参考基因组路径
# REF_GENOME_PATH = "/data/gyy/Data/hg19.fa/hg19.fa"
#
# # 2. VCF 文件列表
# VCF_CONFIG = {
#     'training': "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC/COSMIC_training.vcf",
#     'testing': "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC/COSMIC_testing.vcf",
#     'EOSM': "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/EOSM/EOSM.vcf",
#     'SomaMutDB': "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/SomaMutDBpart/SomaMutDB.vcf"
# }
#
# # 3. 输出目录
# SAVE_DIR = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/one-hot/"
#
# # 4. 序列窗口大小
# # 以前的代码使用了 101bp 长度。
# SEQUENCE_LENGTH = 101
# FLANK_LEFT = 50  # 突变位点左侧取 50个
# # 突变位点本身 1 个
# FLANK_RIGHT = 50  # 突变位点右侧取 50 个
#
# # ==========================================================
#
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)
#
#
# def get_onehot_map():
#     """定义 One-Hot 映射"""
#     # shape: (4,)
#     return {
#         'A': np.array([1, 0, 0, 0], dtype=np.float32),
#         'C': np.array([0, 1, 0, 0], dtype=np.float32),
#         'G': np.array([0, 0, 1, 0], dtype=np.float32),
#         'T': np.array([0, 0, 0, 1], dtype=np.float32),
#         'N': np.array([0, 0, 0, 0], dtype=np.float32),  # 未知碱基
#         'a': np.array([1, 0, 0, 0], dtype=np.float32),
#         'c': np.array([0, 1, 0, 0], dtype=np.float32),
#         'g': np.array([0, 0, 1, 0], dtype=np.float32),
#         't': np.array([0, 0, 0, 1], dtype=np.float32)
#     }
#
#
# def generate_onehot(fasta_path, vcf_path, mode_name):
#     print(f"\n>>> Processing {mode_name} ...")
#
#     if not os.path.exists(fasta_path):
#         raise FileNotFoundError(f"Reference Genome not found at: {fasta_path}")
#
#     # 1. 打开 Fasta 和 VCF
#     fasta = pysam.FastaFile(fasta_path)
#     try:
#         df = pd.read_csv(vcf_path, sep='\t')
#     except Exception as e:
#         print(f"  Error reading VCF {vcf_path}: {e}")
#         return
#
#     onehot_map = get_onehot_map()
#     encoded_list = []
#
#     # 2. 遍历 VCF
#     # 确保列名正确 (兼容 chr, pos, ref, alt 大小写)
#     df.columns = [c.lower() for c in df.columns]
#
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Encoding {mode_name}"):
#         chrom = str(row['chr'])
#         pos = int(row['pos'])  # VCF 是 1-based
#
#         # 兼容 chr 前缀 (hg19 通常需要 chr, 但有的 vcf 没有)
#         # 你的 fasta 如果是 chr1, chr2... 而 vcf 是 1, 2...
#         if not chrom.startswith('chr'):
#             chrom_query = f"chr{chrom}"
#         else:
#             chrom_query = chrom
#
#         # 检查染色体是否存在于 fasta 中
#         if chrom_query not in fasta.references:
#             # 尝试另一种格式 (去掉 chr)
#             if chrom.startswith('chr'):
#                 chrom_query = chrom[3:]
#
#             if chrom_query not in fasta.references:
#                 # 实在找不到，填全0
#                 encoded_list.append(np.zeros((SEQUENCE_LENGTH, 4), dtype=np.float32))
#                 continue
#
#
#         mut_idx_0based = pos - 1
#         start = mut_idx_0based - FLANK_LEFT
#         end = mut_idx_0based + 1 + FLANK_RIGHT
#
#         try:
#             seq_str = fasta.fetch(chrom_query, start, end).upper()
#         except:
#             # 越界等情况
#             seq_str = "N" * SEQUENCE_LENGTH
#
#         # 长度校验
#         if len(seq_str) != SEQUENCE_LENGTH:
#             # 如果在染色体边缘，可能长度不够，进行 padding
#             if len(seq_str) < SEQUENCE_LENGTH:
#                 seq_str = seq_str.ljust(SEQUENCE_LENGTH, 'N')
#             else:
#                 seq_str = seq_str[:SEQUENCE_LENGTH]
#
#         # 4. One-Hot 编码
#         # shape: (100, 4)
#         mat = []
#         for base in seq_str:
#             mat.append(onehot_map.get(base, onehot_map['N']))
#
#         mat = np.array(mat, dtype=np.float32)  # (100, 4)
#
#         # 转换为 (4, 100) 以符合 PyTorch 习惯 (Channels, Seq_Len)，或者保持 (100, 4) 并在 dataset 里展平
#         # 为了方便 flatten，保持 (100, 4) 或者 (4, 100) 都可以。
#         # 这里我们保存为 (4, 100)，即 (A, C, G, T) 四个通道
#         mat = mat.transpose(1, 0)  # -> (4, 100)
#
#         encoded_list.append(mat)
#
#     # 5. 堆叠并保存
#     # Shape: (N, 4, 100)
#     final_array = np.stack(encoded_list, axis=0)
#
#     save_name = f"onehot_{mode_name}.npy"
#     save_path = os.path.join(SAVE_DIR, save_name)
#     np.save(save_path, final_array)
#
#     print(f"  Saved {mode_name}: {final_array.shape} -> {save_path}")
#
#
# # ==========================================================
# if __name__ == "__main__":
#     print(f"Target Directory: {SAVE_DIR}")
#     print(f"Reference Genome: {REF_GENOME_PATH}")
#
#     if not os.path.exists(REF_GENOME_PATH):
#         print(" 错误：请先修改脚本中的 REF_GENOME_PATH 为真实的 hg19.fa 路径！")
#     else:
#         for mode, vpath in VCF_CONFIG.items():
#             if os.path.exists(vpath):
#                 generate_onehot(REF_GENOME_PATH, vpath, mode)
#             else:
#                 print(f"Skipping {mode} (VCF not found: {vpath})")