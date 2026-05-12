import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import pysam  # 需要安装: pip install pysam
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ================= 配置 =================
# 模型路径
MODEL_PATH = "/data/gyy/Project/Feature-Tool/DNABERT-2-117M/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# 参考基因组路径
REF_GENOME = "/data/gyy/Data/hg19.fa/hg19.fa"

# ！！！修改点 1：更新输入文件路径！！！
INPUTS = {
    "KRAS": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/model_input/KRAS_hg19_final.vcf",
    "LUSC": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/model_input/LUSC_hg19_final.vcf",
    "TP53": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/model_input/TP53_hg19_final.vcf"
}

# ！！！修改点 2：更新输出保存路径到 case_analysis 目录下！！！
SAVE_DIR = "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/feature/DNAbert-2/"
os.makedirs(SAVE_DIR, exist_ok=True)

# 序列提取参数
RADIUS = 50


def get_sequences_from_txt(txt_path, ref_fasta_path):
    """
    读取 txt/vcf 文件，根据 chr 和 pos 从参考基因组提取序列
    """
    print(f"Loading coordinates from {txt_path}...")

    try:
        # ！！！修改点 3：使用 \s+ 作为分隔符，兼容空格和制表符！！！
        df = pd.read_csv(txt_path, sep=r'\s+')
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return []

    # 确保列名处理 (去除可能存在的空格，并转小写以防万一)
    df.columns = [c.strip().lower() for c in df.columns]

    # 检查必要的列
    if 'chr' not in df.columns or 'pos' not in df.columns:
        print(f"Error: 'chr' or 'pos' column missing in {txt_path}")
        print(f"Columns found: {df.columns}")
        return []

    print(f"Opening Reference Genome: {ref_fasta_path}...")
    fasta = pysam.FastaFile(ref_fasta_path)

    seqs = []
    valid_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Seqs"):
        chrom = str(row['chr'])
        pos = int(row['pos'])

        # 处理染色体命名不一致问题
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"

        # 处理特殊的 MT / M
        if chrom == "chrMT": chrom = "chrM"

        # 检查染色体是否存在于 reference 中
        if chrom not in fasta.references:
            if chrom.replace("chr", "") in fasta.references:
                chrom = chrom.replace("chr", "")
            else:
                continue

        # 计算提取范围
        start = (pos - 1) - RADIUS
        end = (pos - 1) + RADIUS

        # 边界检查
        if start < 0: start = 0

        try:
            # 提取序列并转大写
            seq = fasta.fetch(chrom, start, end).upper()

            # 长度不够时补 N
            if len(seq) < (RADIUS * 2):
                seq = seq.ljust(RADIUS * 2, 'N')

            seqs.append(seq)
            valid_count += 1

        except KeyError:
            print(f"Warning: Coordinate error at {chrom}:{pos}")
            continue

    fasta.close()
    print(f"Extracted {valid_count} sequences from {len(df)} rows.")
    return seqs


def extract_features():
    print(f"Loading Model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    BATCH_SIZE = 32
    TARGET_LEN = 100  # 输出特征的固定长度

    for name, txt_path in INPUTS.items():
        print(f"\n>>> Processing {name}...")

        seqs = get_sequences_from_txt(txt_path, REF_GENOME)

        if len(seqs) == 0:
            print(f"Skipping {name} (No sequences found).")
            continue

        feats = []
        for i in tqdm(range(0, len(seqs), BATCH_SIZE), desc="Inference"):
            batch = seqs[i: i + BATCH_SIZE]

            inputs = tokenizer(batch, return_tensors="pt", padding="max_length", max_length=150, truncation=True).to(
                DEVICE)

            with torch.no_grad():
                out = model(**inputs)[0]
                out = out.transpose(1, 2)
                resized = F.interpolate(out, size=TARGET_LEN, mode='linear', align_corners=False)
                feats.append(resized.cpu().numpy())

        final_feat = np.concatenate(feats, axis=0)

        save_path = os.path.join(SAVE_DIR, f"{name}_dnabert2.npy")
        np.save(save_path, final_feat)
        print(f"  Saved to: {save_path} | Shape: {final_feat.shape}")


if __name__ == "__main__":
    extract_features()





# #gen_dnabert_test2.py代码
# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel
#
# # ================= 配置 =================
# # 模型路径 (请确认路径正确)
# MODEL_PATH = "/data/gyy/Project/Feature-Tool/DNABERT-2-117M/"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# os.environ["USE_TF"] = "0"
# os.environ["USE_TORCH"] = "1"
# # 输入 FASTA 路径
# INPUTS = {
#     "EOSM": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/EOSM.fasta",
#     "SomaMutDB": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/SomaMutDB.fasta"
# }
#
# # 输出保存路径
# SAVE_DIR = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAbert-2/"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
#
# def read_fasta(path):
#     seqs = []
#     print(f"Reading {path}...")
#     with open(path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line.startswith(">"):
#                 seqs.append(line.upper())
#     return seqs
#
#
# def extract_features():
#     print(f"Loading Model from {MODEL_PATH}...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
#     model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
#
#     BATCH_SIZE = 32
#     TARGET_LEN = 100  # 模型要求的固定长度
#
#     for name, fasta_path in INPUTS.items():
#         print(f"\n>>> Processing {name}...")
#         seqs = read_fasta(fasta_path)
#         print(f"    Count: {len(seqs)}")
#
#         feats = []
#         for i in tqdm(range(0, len(seqs), BATCH_SIZE)):
#             batch = seqs[i: i + BATCH_SIZE]
#             inputs = tokenizer(batch, return_tensors="pt", padding="max_length", max_length=150, truncation=True).to(
#                 DEVICE)
#
#             with torch.no_grad():
#                 # [Batch, Len, 768]
#                 out = model(**inputs)[0]
#                 # 转置为 [Batch, 768, Len] 以便插值
#                 out = out.transpose(1, 2)
#                 # 线性插值到固定长度 100
#                 resized = F.interpolate(out, size=TARGET_LEN, mode='linear', align_corners=False)
#                 # 转回 numpy [Batch, 768, 100]
#                 feats.append(resized.cpu().numpy())
#
#         final_feat = np.concatenate(feats, axis=0)
#
#         # 保存文件名: EOSM_dnabert2.npy
#         save_path = os.path.join(SAVE_DIR, f"{name}_dnabert2.npy")
#         np.save(save_path, final_feat)
#         print(f"    ✅ Saved to: {save_path} | Shape: {final_feat.shape}")
#
#
# if __name__ == "__main__":
#     extract_features()