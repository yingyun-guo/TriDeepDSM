# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys

# ================= 配置区域 =================
# 1. iFeatureOmega 路径 (请确保路径正确)
IFEATURE_PATH = "/data/gyy/Project/Feature-Tool/iFeatureOmega-CLI"
sys.path.append(IFEATURE_PATH)
try:
    from iFeatureOmegaCLI import iDNA
except ImportError:
    print(" Error: 无法导入 iFeatureOmegaCLI，请检查路径配置！")
    sys.exit(1)

# 2. 参考基因组
HG19_PATH = "/data/gyy/Data/hg19.fa/hg19.fa"
PARAM_FILE = os.path.join(IFEATURE_PATH, "parameters/DNA_parameters_setting.json")

# 3. 输入数据集配置 (已修改为 KRAS, LUSC, TP53)
DATASETS = [
    {
        "name": "KRAS",
        "path": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/model_input/KRAS_hg19_final.vcf",
        "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/feature/physicochemical/KRAS"
    },
    {
        "name": "LUSC",
        "path": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/model_input/LUSC_hg19_final.vcf",
        "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/feature/physicochemical/LUSC"
    },
    {
        "name": "TP53",
        "path": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/model_input/TP53_hg19_final.vcf",
        "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/plot/case_analysis/data/feature/physicochemical/TP53"
    }
]

# 4. 序列提取参数 (保持原代码设定: 11bp)
FLANK = 5
TOTAL_LEN = 2 * FLANK + 1  # 11bp

# 5. 特征列表
FEATURES = [
    "Kmer type 1", "NAC", "CKSNAP type 1",
    "Mismatch", "MMI", "NMBroto",
    "Z_curve_9bit", "Z_curve_12bit", "Z_curve_36bit",
    "Z_curve_48bit", "Z_curve_144bit"
]

# ===========================================

if not os.path.exists(HG19_PATH):
    raise FileNotFoundError(f"参考基因组文件不存在: {HG19_PATH}")

print(" 正在加载 hg19 基因组 (这可能需要几分钟)...")
# 加载全基因组到内存 (Bio.SeqIO)
genome = SeqIO.to_dict(SeqIO.parse(HG19_PATH, "fasta"))
print("基因组加载完成！")


def extract_sequence(chrom, pos, ref, alt):
    """提取 Ref 和 Mut 序列"""
    key = str(chrom)
    # 染色体命名兼容性处理
    if key not in genome:
        if f"chr{key}" in genome:
            key = f"chr{key}"
        elif key.startswith("chr") and key[3:] in genome:
            key = key[3:]
        else:
            return None, None

    seq = genome[key].seq
    center = pos - 1
    start = center - FLANK
    end = center + FLANK + 1

    segment = seq[max(start, 0): min(end, len(seq))]
    segment = str(segment).upper()

    # 边界填充
    left = max(0, -start)
    right = TOTAL_LEN - len(segment) - left
    ref_seq = "A" * left + segment + "A" * right
    ref_seq = ''.join(b if b in 'ACGT' else 'A' for b in ref_seq)

    # 构建突变序列
    mut_seq = list(ref_seq)
    if 0 <= FLANK < len(mut_seq):
        mut_seq[FLANK] = alt.upper()
    mut_seq = "".join(mut_seq)

    return ref_seq, mut_seq


def get_author_filename(feat_name, seq_type, suffix_name):
    """
    生成对齐作者风格的文件名
    """
    base = feat_name.replace(" type 1", "_type_1").replace(" ", "_")

    # 构建文件名: {state}_{feature}_{dataset}.txt
    if seq_type == "diff":
        return f"diffe_{base}_{suffix_name}.txt"
    elif seq_type == "normal":
        return f"normal_{base}_{suffix_name}.txt"
    elif seq_type == "mutation":
        return f"mutation_{base}_{suffix_name}.txt"
    return f"feature{base}_{suffix_name}.txt"


def run_ifeature_wrapper(fasta_path, feat_name, out_path):
    """调用 iFeatureOmega 计算特征"""
    dna = iDNA(fasta_path)
    try:
        dna.import_parameters(PARAM_FILE)
    except:
        pass

    # 参数修正
    if feat_name == "Kmer type 1":
        dna.k = 2
        dna.normalize = True
    elif "CKSNAP" in feat_name:
        dna.kspace = 3
        dna.normalize = True
    elif feat_name == "MMI":
        dna.nlag = 3
        dna.normalize = True
    elif feat_name == "NMBroto":
        dna.nlag = 3
        dna.normalize = True
    elif "Z_curve" in feat_name:
        bit = feat_name.split("_")[-1]
        dna.zcurve = bit

    try:
        dna.get_descriptor(feat_name)
        if hasattr(dna, 'encodings') and dna.encodings is not None and len(dna.encodings) > 0:
            dna.to_csv(out_path, index=False, header=True)
            return True
        else:
            print(f"    [Warn] {feat_name} 返回空")
            return False
    except Exception as e:
        print(f"    [Error] {feat_name}: {e}")
        return False


def calculate_manual_RCKmer(fasta_path, out_dir, seq_type, suffix_name):
    """手动计算 RCKmer"""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    rc_records = [SeqRecord(rec.seq.reverse_complement(), id=rec.id, description="") for rec in records]

    temp_rc_fasta = os.path.join(out_dir, f"temp_rc_{seq_type}.fasta")
    SeqIO.write(rc_records, temp_rc_fasta, "fasta")

    out_name = os.path.join(out_dir, f"{seq_type}_RCKmer_type_1_{suffix_name}.txt")

    dna = iDNA(temp_rc_fasta)
    try:
        dna.import_parameters(PARAM_FILE)
    except:
        pass
    dna.k = 2
    dna.normalize = True

    try:
        dna.get_descriptor("Kmer type 1")
        if dna.encodings is not None:
            dna.encodings.columns = [c.replace("2mer", "RC2mer_type_1") for c in dna.encodings.columns]
            dna.to_csv(out_name, index=False, header=True)
    except Exception as e:
        print(f"    Failed RCKmer: {e}")

    if os.path.exists(temp_rc_fasta):
        os.remove(temp_rc_fasta)


def process_dataset(config):
    name = config['name']
    path = config['path']
    out_dir = config['out_dir']

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"正在处理: {name}")
    print(f"{'=' * 60}")

    # 1. 读取 VCF 数据 (兼容多空格/制表符)
    try:
        df = pd.read_csv(path, sep=r'\s+', comment='#')
        df.columns = [c.strip().lower() for c in df.columns]  # 统一小写
    except Exception as e:
        print(f"读取失败 {path}: {e}")
        return

    # 检查列
    if not all(col in df.columns for col in ['chr', 'pos', 'ref', 'alt']):
        print(f"缺少必要列 (chr, pos, ref, alt). 当前列: {df.columns}")
        return

    # 2. 提取序列
    records = {"normal": [], "mutation": []}
    skipped = 0

    for idx, row in df.iterrows():
        chrom = str(row['chr'])
        pos = int(row['pos'])
        ref = str(row['ref'])
        alt = str(row['alt'])

        ref_seq, mut_seq = extract_sequence(chrom, pos, ref, alt)

        if ref_seq is None:
            skipped += 1
            continue

        seq_id = f"{chrom}_{pos}_{ref}to{alt}"
        records["normal"].append(SeqRecord(Seq(ref_seq), id=f"{seq_id}_ref", description=""))
        records["mutation"].append(SeqRecord(Seq(mut_seq), id=f"{seq_id}_alt", description=""))

    print(f"  序列提取完成: {len(records['normal'])} 条 (跳过 {skipped} 条无效坐标)")

    # 3. 保存临时 FASTA 并计算特征
    for seq_type in ["normal", "mutation"]:
        fasta_file = os.path.join(out_dir, f"{name}_{seq_type}.fasta")
        SeqIO.write(records[seq_type], fasta_file, "fasta")

        print(f"  > 计算 {seq_type} 特征...")
        # 常规特征
        for feat in FEATURES:
            filename = get_author_filename(feat, seq_type, name)
            out_file = os.path.join(out_dir, filename)
            # 只有文件不存在时才计算，节省时间
            if not os.path.exists(out_file):
                run_ifeature_wrapper(fasta_file, feat, out_file)

        # RCKmer
        calculate_manual_RCKmer(fasta_file, out_dir, seq_type, name)

    # 4. 计算 Diff (Mutation - Normal)
    print(f"  > 计算 Diff 特征...")

    # 包含 RCKmer
    all_feats = FEATURES + ["RCKmer type 1"]

    for feat in all_feats:
        # 获取文件名
        if feat == "RCKmer type 1":
            norm_file = os.path.join(out_dir, f"normal_RCKmer_type_1_{name}.txt")
            mut_file = os.path.join(out_dir, f"mutation_RCKmer_type_1_{name}.txt")
            diff_file = os.path.join(out_dir, f"diffe_RCKmer_type_1_{name}.txt")
        else:
            norm_file = os.path.join(out_dir, get_author_filename(feat, "normal", name))
            mut_file = os.path.join(out_dir, get_author_filename(feat, "mutation", name))
            diff_file = os.path.join(out_dir, get_author_filename(feat, "diff", name))

        if not (os.path.exists(norm_file) and os.path.exists(mut_file)):
            continue

        try:
            df_n = pd.read_csv(norm_file)
            df_m = pd.read_csv(mut_file)

            # 只选取数值列进行相减
            cols = df_n.select_dtypes(include=['number']).columns
            df_diff = df_m[cols] - df_n[cols]

            df_diff.to_csv(diff_file, sep="\t", index=False)
        except Exception as e:
            print(f"    Failed diff {feat}: {e}")

    print(f" {name} 处理完毕！")


if __name__ == "__main__":
    for conf in DATASETS:
        process_dataset(conf)
    print("\n所有任务完成！")




# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# import sys
#
# # ================= 配置区域 =================
# # 1. iFeatureOmega 路径 (请确保路径正确)
# IFEATURE_PATH = "/data/gyy/Project/Feature-Tool/iFeatureOmega-CLI"
# sys.path.append(IFEATURE_PATH)
# try:
#     from iFeatureOmegaCLI import iDNA
# except ImportError:
#     print(" Error: 无法导入 iFeatureOmegaCLI，请检查路径配置！")
#     sys.exit(1)
#
# # 2. 参考基因组
# HG19_PATH = "/data/gyy/Data/hg19.fa/hg19.fa"
# PARAM_FILE = os.path.join(IFEATURE_PATH, "parameters/DNA_parameters_setting.json")
#
# # 3. 输入数据集配置 (Name -> Path)
# DATASETS = [
#     {
#         "name": "MFDSMC_train",
#         "path": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/MFDSMC_train.txt",
#         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/MFDSMC"
#     },
#     {
#         "name": "EPEL_train",
#         "path": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/EPEL_train.txt",
#         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/EPEL"
#     }
# ]
#
# # DATASETS = [
# #     {
# #         "name": "MFDSMC_test1",
# #         "path": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/MFDSMC_test1.txt",
# #         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/MFDSMC"
# #     },
# #     {
# #         "name": "MFDSMC_test2",
# #         "path": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/MFDSMC_test2.txt",
# #         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/MFDSMC"
# #     },
# #     {
# #         "name": "EPEL_test",
# #         "path": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/EPEL_test.txt",
# #         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/EPEL"
# #     }
# # ]
#
# # 4. 序列提取参数 (保持原代码设定: 11bp)
# FLANK = 5
# TOTAL_LEN = 2 * FLANK + 1  # 11bp
#
# # 5. 特征列表
# FEATURES = [
#     "Kmer type 1", "NAC", "CKSNAP type 1",
#     "Mismatch", "MMI", "NMBroto",
#     "Z_curve_9bit", "Z_curve_12bit", "Z_curve_36bit",
#     "Z_curve_48bit", "Z_curve_144bit"
# ]
#
# # ===========================================
#
# if not os.path.exists(HG19_PATH):
#     raise FileNotFoundError(f"参考基因组文件不存在: {HG19_PATH}")
#
# print(" 正在加载 hg19 基因组 (这可能需要几分钟)...")
# # 加载全基因组到内存 (Bio.SeqIO)
# genome = SeqIO.to_dict(SeqIO.parse(HG19_PATH, "fasta"))
# print("基因组加载完成！")
#
#
# def extract_sequence(chrom, pos, ref, alt):
#     """提取 Ref 和 Mut 序列"""
#     key = str(chrom)
#     # 染色体命名兼容性处理
#     if key not in genome:
#         if f"chr{key}" in genome:
#             key = f"chr{key}"
#         elif key.startswith("chr") and key[3:] in genome:
#             key = key[3:]
#         else:
#             return None, None
#
#     seq = genome[key].seq
#     center = pos - 1
#     start = center - FLANK
#     end = center + FLANK + 1
#
#     segment = seq[max(start, 0): min(end, len(seq))]
#     segment = str(segment).upper()
#
#     # 边界填充
#     left = max(0, -start)
#     right = TOTAL_LEN - len(segment) - left
#     ref_seq = "A" * left + segment + "A" * right
#     ref_seq = ''.join(b if b in 'ACGT' else 'A' for b in ref_seq)
#
#     # 构建突变序列
#     mut_seq = list(ref_seq)
#     if 0 <= FLANK < len(mut_seq):
#         mut_seq[FLANK] = alt.upper()
#     mut_seq = "".join(mut_seq)
#
#     return ref_seq, mut_seq
#
#
# def get_author_filename(feat_name, seq_type, suffix_name):
#     """
#     生成对齐作者风格的文件名
#     e.g., diffe_Kmer_type_1_MFDSMC_test1.txt
#     """
#     base = feat_name.replace(" type 1", "_type_1").replace(" ", "_")
#
#     # 构建文件名: {state}_{feature}_{dataset}.txt
#     if seq_type == "diff":
#         return f"diffe_{base}_{suffix_name}.txt"
#     elif seq_type == "normal":
#         return f"normal_{base}_{suffix_name}.txt"
#     elif seq_type == "mutation":
#         return f"mutation_{base}_{suffix_name}.txt"
#     return f"feature{base}_{suffix_name}.txt"
#
#
# def run_ifeature_wrapper(fasta_path, feat_name, out_path):
#     """调用 iFeatureOmega 计算特征"""
#     dna = iDNA(fasta_path)
#     try:
#         dna.import_parameters(PARAM_FILE)
#     except:
#         pass
#
#     # 参数修正
#     if feat_name == "Kmer type 1":
#         dna.k = 2
#         dna.normalize = True
#     elif "CKSNAP" in feat_name:
#         dna.kspace = 3
#         dna.normalize = True
#     elif feat_name == "MMI":
#         dna.nlag = 3
#         dna.normalize = True
#     elif feat_name == "NMBroto":
#         dna.nlag = 3
#         dna.normalize = True
#     elif "Z_curve" in feat_name:
#         bit = feat_name.split("_")[-1]
#         dna.zcurve = bit
#
#     try:
#         dna.get_descriptor(feat_name)
#         if hasattr(dna, 'encodings') and dna.encodings is not None and len(dna.encodings) > 0:
#             dna.to_csv(out_path, index=False, header=True)
#             return True
#         else:
#             print(f"    [Warn] {feat_name} 返回空")
#             return False
#     except Exception as e:
#         print(f"    [Error] {feat_name}: {e}")
#         return False
#
#
# def calculate_manual_RCKmer(fasta_path, out_dir, seq_type, suffix_name):
#     """手动计算 RCKmer"""
#     # print(f"    > 手动生成 RCKmer ({seq_type})...")
#     records = list(SeqIO.parse(fasta_path, "fasta"))
#     rc_records = [SeqRecord(rec.seq.reverse_complement(), id=rec.id, description="") for rec in records]
#
#     temp_rc_fasta = os.path.join(out_dir, f"temp_rc_{seq_type}.fasta")
#     SeqIO.write(rc_records, temp_rc_fasta, "fasta")
#
#     out_name = os.path.join(out_dir, f"{seq_type}_RCKmer_type_1_{suffix_name}.txt")
#
#     dna = iDNA(temp_rc_fasta)
#     try:
#         dna.import_parameters(PARAM_FILE)
#     except:
#         pass
#     dna.k = 2
#     dna.normalize = True
#
#     try:
#         dna.get_descriptor("Kmer type 1")
#         if dna.encodings is not None:
#             dna.encodings.columns = [c.replace("2mer", "RC2mer_type_1") for c in dna.encodings.columns]
#             dna.to_csv(out_name, index=False, header=True)
#             # print(f"    Success RCKmer -> {os.path.basename(out_name)}")
#     except Exception as e:
#         print(f"    Failed RCKmer: {e}")
#
#     if os.path.exists(temp_rc_fasta):
#         os.remove(temp_rc_fasta)
#
#
# def process_dataset(config):
#     name = config['name']
#     path = config['path']
#     out_dir = config['out_dir']
#
#     os.makedirs(out_dir, exist_ok=True)
#
#     print(f"\n{'=' * 60}")
#     print(f"正在处理: {name}")
#     print(f"{'=' * 60}")
#
#     # 1. 读取 TXT 数据
#     try:
#         df = pd.read_csv(path, sep='\t')
#         df.columns = [c.strip().lower() for c in df.columns]  # 统一小写
#     except Exception as e:
#         print(f"读取失败 {path}: {e}")
#         return
#
#     # 检查列
#     if not all(col in df.columns for col in ['chr', 'pos', 'ref', 'alt']):
#         print(f"缺少必要列 (chr, pos, ref, alt). 当前列: {df.columns}")
#         return
#
#     # 2. 提取序列
#     records = {"normal": [], "mutation": []}
#     skipped = 0
#
#     for idx, row in df.iterrows():
#         chrom = str(row['chr'])
#         pos = int(row['pos'])
#         ref = str(row['ref'])
#         alt = str(row['alt'])
#
#         ref_seq, mut_seq = extract_sequence(chrom, pos, ref, alt)
#
#         if ref_seq is None:
#             skipped += 1
#             continue
#
#         seq_id = f"{chrom}_{pos}_{ref}to{alt}"
#         records["normal"].append(SeqRecord(Seq(ref_seq), id=f"{seq_id}_ref", description=""))
#         records["mutation"].append(SeqRecord(Seq(mut_seq), id=f"{seq_id}_alt", description=""))
#
#     print(f"  序列提取完成: {len(records['normal'])} 条 (跳过 {skipped} 条无效坐标)")
#
#     # 3. 保存临时 FASTA 并计算特征
#     for seq_type in ["normal", "mutation"]:
#         fasta_file = os.path.join(out_dir, f"{name}_{seq_type}.fasta")
#         SeqIO.write(records[seq_type], fasta_file, "fasta")
#
#         print(f"  > 计算 {seq_type} 特征...")
#         # 常规特征
#         for feat in FEATURES:
#             filename = get_author_filename(feat, seq_type, name)
#             out_file = os.path.join(out_dir, filename)
#             # 只有文件不存在时才计算，节省时间
#             if not os.path.exists(out_file):
#                 run_ifeature_wrapper(fasta_file, feat, out_file)
#
#         # RCKmer
#         calculate_manual_RCKmer(fasta_file, out_dir, seq_type, name)
#
#     # 4. 计算 Diff (Mutation - Normal)
#     print(f"  > 计算 Diff 特征...")
#
#     # 包含 RCKmer
#     all_feats = FEATURES + ["RCKmer type 1"]
#
#     for feat in all_feats:
#         # 获取文件名
#         if feat == "RCKmer type 1":
#             norm_file = os.path.join(out_dir, f"normal_RCKmer_type_1_{name}.txt")
#             mut_file = os.path.join(out_dir, f"mutation_RCKmer_type_1_{name}.txt")
#             diff_file = os.path.join(out_dir, f"diffe_RCKmer_type_1_{name}.txt")
#         else:
#             norm_file = os.path.join(out_dir, get_author_filename(feat, "normal", name))
#             mut_file = os.path.join(out_dir, get_author_filename(feat, "mutation", name))
#             diff_file = os.path.join(out_dir, get_author_filename(feat, "diff", name))
#
#         if not (os.path.exists(norm_file) and os.path.exists(mut_file)):
#             continue
#
#         try:
#             df_n = pd.read_csv(norm_file)
#             df_m = pd.read_csv(mut_file)
#
#             # 只选取数值列进行相减
#             cols = df_n.select_dtypes(include=['number']).columns
#             df_diff = df_m[cols] - df_n[cols]
#
#             df_diff.to_csv(diff_file, sep="\t", index=False)
#             # print(f"    生成 -> {os.path.basename(diff_file)}")
#         except Exception as e:
#             print(f"    Failed diff {feat}: {e}")
#
#     print(f" {name} 处理完毕！")
#
#
# if __name__ == "__main__":
#     for conf in DATASETS:
#         process_dataset(conf)
#     print("\n所有任务完成！")



# # -*- coding: utf-8 -*-
# """
# DeepSTF 特征工程 - 终极版（文件名完全对齐作者风格）
# """
#
# import os
# import pandas as pd
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# import sys
#
# sys.path.append("/data/gyy/Project/Feature-Tool/iFeatureOmega-CLI")
# from iFeatureOmegaCLI import iDNA
#
# hg19_path = "/data/gyy/Data/hg19.fa/hg19.fa"
# param_file = "/data/gyy/Project/Feature-Tool/iFeatureOmega-CLI/parameters/DNA_parameters_setting.json"
#
# if not os.path.exists(hg19_path):
#     raise FileNotFoundError(f"参考基因组文件不存在: {hg19_path}")
#
# print("正在加载 hg19 基因组...")
# genome = SeqIO.to_dict(SeqIO.parse(hg19_path, "fasta"))
#
# FLANK = 5
# TOTAL_LEN = 2 * FLANK + 1  # 11bp
#
# # 特征列表（统一加 type_1 后缀，与作者一致）
# features = [
#     "Kmer type 1", "NAC", "CKSNAP type 1",
#     "Mismatch", "MMI", "NMBroto",
#     "Z_curve_9bit", "Z_curve_12bit", "Z_curve_36bit",
#     "Z_curve_48bit", "Z_curve_144bit"
# ]
#
# # ======== 修改这里切换数据集 ========
# datasets = [
#     # {
#     #     "name": "SomaMutDB",
#     #     "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/SomaMutDBpart/SomaMutDB.vcf",
#     #     "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical-new/SomaMutDB",
#     #     "prefix": ""
#     # },
#     # {
#     #     "name": "EOSM",
#     #     "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/EOSM/EOSM.vcf",
#     #     "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical-new/EOSM",
#     #     "prefix": ""
#     # },
#     {
#         "name": "COSMIC_train",
#         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/COSMIC_training.vcf",
#         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical-new/COSMIC",
#         "prefix": "train_"
#     },
#     {
#         "name": "COSMIC_test",
#         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/COSMIC_testing.vcf",
#         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical-new/COSMIC",
#         "prefix": "test_"
#     },
# ]
#
# for ds in datasets:
#     os.makedirs(ds["out_dir"], exist_ok=True)
#
#
# def extract_sequence(chrom, pos, ref, alt):
#     key = str(chrom)
#     if key not in genome:
#         if f"chr{key}" in genome:
#             key = f"chr{key}"
#         elif key.startswith("chr") and key[3:] in genome:
#             key = key[3:]
#         else:
#             return None, None
#
#     seq = genome[key].seq
#     center = pos - 1
#     start = center - FLANK
#     end = center + FLANK + 1
#
#     segment = seq[max(start, 0): min(end, len(seq))]
#     segment = str(segment).upper()
#
#     left = max(0, -start)
#     right = TOTAL_LEN - len(segment) - left
#     ref_seq = "A" * left + segment + "A" * right
#     ref_seq = ''.join(b if b in 'ACGT' else 'A' for b in ref_seq)
#
#     mut_seq = list(ref_seq)
#     if 0 <= FLANK < len(mut_seq):
#         mut_seq[FLANK] = alt.upper()
#     mut_seq = "".join(mut_seq)
#
#     return ref_seq, mut_seq
#
#
# def get_author_filename(feat_name, seq_type, prefix=""):
#     """生成完全对齐作者的文件名"""
#     base = feat_name.replace(" type 1", "_type_1").replace(" ", "_")
#     if seq_type == "diff":
#         return f"{prefix}diffe_{base}.txt"
#     elif seq_type == "normal":
#         return f"{prefix}normal_{base}.txt"
#     elif seq_type == "mutation":
#         return f"{prefix}mutation_{base}.txt"
#     return f"{prefix}feature{base}.txt"
#
#
# def run_ifeature_wrapper(fasta_path, feat_name, out_path):
#     dna = iDNA(fasta_path)
#     try:
#         dna.import_parameters(param_file)
#     except:
#         pass
#
#     if feat_name == "Kmer type 1":
#         dna.k = 2
#         dna.normalize = True
#     elif "CKSNAP" in feat_name:
#         dna.kspace = 3
#         dna.normalize = True
#     elif feat_name == "MMI":
#         dna.nlag = 3
#         dna.normalize = True
#     elif feat_name == "NMBroto":
#         dna.nlag = 3
#         dna.normalize = True
#     elif "Z_curve" in feat_name:
#         bit = feat_name.split("_")[-1]
#         dna.zcurve = bit
#
#     try:
#         dna.get_descriptor(feat_name)
#         if hasattr(dna, 'encodings') and dna.encodings is not None and len(dna.encodings) > 0:
#             dna.to_csv(out_path, index=False, header=True)
#             return True
#         else:
#             print(f"  [Warn] {feat_name} 返回空")
#             return False
#     except Exception as e:
#         print(f"  [Error] {feat_name}: {e}")
#         return False
#
#
# def calculate_manual_RCKmer(fasta_path, out_dir, seq_type, prefix=""):
#     print(f"  > 手动生成 RCKmer ({seq_type})...")
#     records = list(SeqIO.parse(fasta_path, "fasta"))
#     rc_records = [SeqRecord(rec.seq.reverse_complement(), id=rec.id, description="") for rec in records]
#
#     temp_rc_fasta = os.path.join(out_dir, f"temp_rc_{seq_type}.fasta")
#     SeqIO.write(rc_records, temp_rc_fasta, "fasta")
#
#     out_name = os.path.join(out_dir, f"{prefix}{seq_type}_RCKmer_type_1.txt")
#
#     dna = iDNA(temp_rc_fasta)
#     try:
#         dna.import_parameters(param_file)
#     except:
#         pass
#     dna.k = 2
#     dna.normalize = True
#
#     try:
#         dna.get_descriptor("Kmer type 1")
#         if dna.encodings is not None:
#             dna.encodings.columns = [c.replace("2mer", "RC2mer_type_1") for c in dna.encodings.columns]
#             dna.to_csv(out_name, index=False, header=True)
#             print(f"  Success → {os.path.basename(out_name)}")
#     except Exception as e:
#         print(f"  Failed RCKmer: {e}")
#
#     os.remove(temp_rc_fasta)
#
#
# def process_dataset(name, vcf_path, out_dir, prefix=""):
#     print(f"\n{'='*60}")
#     print(f"正在处理: {prefix}{name}")
#     print(f"{'='*60}")
#
#     # VCF 读取（同之前）
#     try:
#         df = pd.read_csv(vcf_path, sep=r"\s+", header=None, comment='#', dtype=str)
#         if not (df.iloc[0, 0].startswith("chr") or str(df.iloc[0, 0]).isdigit()):
#             df = pd.read_csv(vcf_path, sep=r"\s+", dtype=str)
#         df[1] = pd.to_numeric(df[1], errors='coerce')
#         df = df.dropna(subset=[1])
#         vals = df.values
#     except:
#         df = pd.read_csv(vcf_path, sep=r"\s+", dtype=str)
#         vals = df.values
#
#     records = {"normal": [], "mutation": []}
#     for row in vals:
#         chrom, pos, ref, alt = row[0], int(row[1]), row[2], row[3]
#         ref_seq, mut_seq = extract_sequence(chrom, pos, ref, alt)
#         if ref_seq is None: continue
#         seq_id = f"{chrom}_{pos}_{ref}to{alt}"
#         records["normal"].append(SeqRecord(Seq(ref_seq), id=f"{seq_id}_ref", description=""))
#         records["mutation"].append(SeqRecord(Seq(mut_seq), id=f"{seq_id}_alt", description=""))
#
#     print(f"序列生成完成: {len(records['normal'])} 条")
#
#     for st in ["normal", "mutation"]:
#         SeqIO.write(records[st], os.path.join(out_dir, f"{prefix}{st}_{name}.fasta"), "fasta")
#
#     for seq_type in ["normal", "mutation"]:
#         fasta_file = os.path.join(out_dir, f"{prefix}{seq_type}_{name}.fasta")
#         for feat in features:
#             filename = get_author_filename(feat, seq_type, prefix)
#             out_file = os.path.join(out_dir, filename)
#             run_ifeature_wrapper(fasta_file, feat, out_file)
#
#         calculate_manual_RCKmer(fasta_file, out_dir, seq_type, prefix)
#
#     # 计算 Diff
#     print(f"\n正在计算 {prefix}diff 特征...")
#     for feat in features:
#         base_name = get_author_filename(feat, "", "")  # 无前缀无类型
#         norm_file = os.path.join(out_dir, f"{prefix}normal_{base_name}.txt")
#         mut_file = os.path.join(out_dir, f"{prefix}mutation_{base_name}.txt")
#         diff_file = os.path.join(out_dir, f"{prefix}diffe_{base_name}.txt")
#
#         if not (os.path.exists(norm_file) and os.path.exists(mut_file)):
#             continue
#
#         try:
#             df_n = pd.read_csv(norm_file, sep='\t' if '\t' in open(norm_file).readline() else ',')
#             df_m = pd.read_csv(mut_file, sep='\t' if '\t' in open(mut_file).readline() else ',')
#             cols = df_n.select_dtypes(include=['number']).columns
#             df_diff = df_m[cols] - df_n[cols]
#             df_diff.to_csv(diff_file, sep="\t", index=False, header=True)
#             print(f"  Success → {os.path.basename(diff_file)}")
#         except Exception as e:
#             print(f"  Failed diff {feat}: {e}")
#
#     # RCKmer 的 diff
#     rc_norm = os.path.join(out_dir, f"{prefix}normal_RCKmer_type_1.txt")
#     rc_mut = os.path.join(out_dir, f"{prefix}mutation_RCKmer_type_1.txt")
#     rc_diff = os.path.join(out_dir, f"{prefix}diffe_RCKmer_type_1.txt")
#     if os.path.exists(rc_norm) and os.path.exists(rc_mut):
#         try:
#             df_n = pd.read_csv(rc_norm)
#             df_m = pd.read_csv(rc_mut)
#             cols = df_n.select_dtypes(include=['number']).columns
#             (df_m[cols] - df_n[cols]).to_csv(rc_diff, sep="\t", index=False, header=True)
#         except Exception as e:
#             print(f"  Failed RCKmer diff: {e}")
#
#     print(f"\n{prefix}{name} 处理完成！\n")
#
#
# if __name__ == "__main__":
#     for ds in datasets:
#         process_dataset(ds["name"], ds["vcf"], ds["out_dir"], ds.get("prefix", ""))
#     print("所有数据集处理完成！文件名已完全对齐作者风格。")



# # -*- coding: utf-8 -*-
# """
# DeepSTF 特征工程 - 终极融合版
# 结构：基于你提供的稳健代码
# 参数：基于作者的 11bp 窗口 (FLANK=5) 和 k=2
# """
#
# import os
# import pandas as pd
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# import sys
# import shutil
#
# # ============ 路径配置 ============
# sys.path.append("/data/gyy/Project/Feature-Tool/iFeatureOmega-CLI")
# from iFeatureOmegaCLI import iDNA
#
# hg19_path = "/data/gyy/Data/hg19.fa/hg19.fa"
# param_file = "/data/gyy/Project/Feature-Tool/iFeatureOmega-CLI/parameters/DNA_parameters_setting.json"
#
# if not os.path.exists(hg19_path):
#     raise FileNotFoundError(f"参考基因组文件不存在: {hg19_path}")
#
# print("正在加载 hg19 基因组...")
# genome = SeqIO.to_dict(SeqIO.parse(hg19_path, "fasta"))
#
# # 【核心修改 1】作者使用的是 11bp 窗口 (NAC=0.09)
# FLANK = 5
# TOTAL_LEN = 2 * FLANK + 1  # 11bp
#
# # 特征列表（移除 RCKmer type 1，改为手动计算）
# features = [
#     "Kmer type 1", "NAC", "CKSNAP type 1",
#     "Mismatch", "MMI", "NMBroto",
#     "Z_curve_9bit", "Z_curve_12bit", "Z_curve_36bit",
#     "Z_curve_48bit", "Z_curve_144bit"
# ]
# datasets = [
#     {
#         "name": "SomaMutDB",
#         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/SomaMutDBpart/SomaMutDB.vcf",
#         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/SomaMutDB"
#     },
#     {
#         "name": "EOSM",
#         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/EOSM/EOSM.vcf",
#         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/EOSM"
#     }
# ]
#
# # datasets = [
# #     {
# #         "name": "COSMIC",
# #         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/COSMIC_training.vcf",
# #         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/train_final"
# #     },
# #     {
# #         "name": "COSMIC",
# #         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/COSMIC_testing.vcf",
# #         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/test_final"
# #     }
# # ]
#
# for ds in datasets:
#     os.makedirs(ds["out_dir"], exist_ok=True)
#
#
# def extract_sequence(chrom, pos, ref, alt):
#     """提取 11bp 序列"""
#     key = str(chrom)
#     if key not in genome:
#         if f"chr{key}" in genome:
#             key = f"chr{key}"
#         elif key.startswith("chr") and key[3:] in genome:
#             key = key[3:]
#         else:
#             return None, None
#
#     seq = genome[key].seq
#     center = pos - 1
#     start = center - FLANK
#     end = center + FLANK + 1
#
#     segment = seq[max(start, 0): min(end, len(seq))]
#     segment = str(segment).upper()
#
#     left = max(0, -start)
#     right = TOTAL_LEN - len(segment) - left
#     ref_seq = "A" * left + segment + "A" * right
#     ref_seq = ''.join(b if b in 'ACGT' else 'A' for b in ref_seq)
#
#     # 构建突变序列
#     mut_seq = list(ref_seq)
#     if 0 <= FLANK < len(mut_seq):
#         mut_seq[FLANK] = alt.upper()
#     mut_seq = "".join(mut_seq)
#
#     return ref_seq, mut_seq
#
#
# def get_author_style_name(feat_raw_name):
#     """将特征名转换为作者的文件名格式"""
#     if feat_raw_name == "Kmer type 1": return "feature2mer"
#     if feat_raw_name == "RCKmer": return "featureRC2mer"
#     if feat_raw_name == "CKSNAP type 1": return "featureCKSNAP"
#
#     clean = feat_raw_name.replace(" ", "_").replace("(", "").replace(")", "")
#     if "Z_curve" in clean: return f"feature{clean}"
#     return f"feature{clean}"
#
#
# def run_ifeature_wrapper(fasta_path, feat_name, out_path):
#     """封装 iFeature 调用，植入参数修正"""
#     dna = iDNA(fasta_path)
#     try:
#         dna.import_parameters(param_file)
#     except:
#         pass
#
#     # 【核心修改 2】强制参数对齐作者
#     if feat_name == "Kmer type 1":
#         dna.k = 2  # 作者使用的是 2-mer (16维)
#         dna.normalize = True
#     elif "CKSNAP" in feat_name:
#         dna.gap = 5
#         dna.normalize = True
#     elif feat_name == "MMI":
#         dna.nlag = 3
#         dna.normalize = True
#     elif "Z_curve" in feat_name:
#         # 提取 9bit, 12bit 等
#         bit = feat_name.split("_")[-1]
#         dna.zcurve = bit
#
#     try:
#         dna.get_descriptor(feat_name)
#         if hasattr(dna, 'encodings') and dna.encodings is not None and len(dna.encodings) > 0:
#             dna.to_csv(out_path, index=False, header=True)
#             return True
#         else:
#             print(f"  [Warn] {feat_name} 返回空")
#             return False
#     except Exception as e:
#         print(f"  [Error] {feat_name}: {e}")
#         return False
#
#
# def calculate_manual_RCKmer(fasta_path, out_dir, seq_type):
#     """【核心修改 3】手动计算 RCKmer"""
#     print(f"  > 手动生成 RCKmer ({seq_type})...")
#
#     # 1. 生成反向互补 FASTA
#     records = list(SeqIO.parse(fasta_path, "fasta"))
#     rc_records = []
#     for rec in records:
#         rc_seq = rec.seq.reverse_complement()
#         rc_records.append(SeqRecord(rc_seq, id=rec.id, description=""))
#
#     temp_rc_fasta = os.path.join(out_dir, f"temp_rc_{seq_type}.fasta")
#     SeqIO.write(rc_records, temp_rc_fasta, "fasta")
#
#     # 2. 调用 Kmer (k=2)
#     out_name = os.path.join(out_dir, f"featureRC2mer_{seq_type}.txt")
#
#     # 这里借用 Kmer type 1 的逻辑，但输入是 RC 序列
#     dna = iDNA(temp_rc_fasta)
#     try:
#         dna.import_parameters(param_file)
#     except:
#         pass
#     dna.k = 2
#     dna.normalize = True
#
#     try:
#         dna.get_descriptor("Kmer type 1")
#         # 修改列名以区分
#         if dna.encodings is not None:
#             # 简单的列名替换
#             dna.encodings.columns = [c.replace("2mer", "RC2mer") for c in dna.encodings.columns]
#             dna.to_csv(out_name, index=False, header=True)
#             print(f"  Success RCKmer → OK")
#     except Exception as e:
#         print(f"  Failed RCKmer: {e}")
#
#     if os.path.exists(temp_rc_fasta):
#         os.remove(temp_rc_fasta)
#
#
# def process_dataset(name, vcf_path, out_dir):
#     print(f"\n{'=' * 60}")
#     print(f"正在处理数据集: {name}  →  {vcf_path}")
#     print(f"{'=' * 60}")
#
#     # VCF 读取 (保留你的逻辑，增加注释行过滤)
#     try:
#         df = pd.read_csv(vcf_path, sep=r"\s+", header=None, comment='#', dtype=str)
#         # 简单判定
#         if not (df.iloc[0, 0].startswith("chr") or df.iloc[0, 0] == "1"):
#             df = pd.read_csv(vcf_path, sep=r"\s+", dtype=str)
#
#         # 过滤掉包含 'pos' 的行
#         df[1] = pd.to_numeric(df[1], errors='coerce')
#         df = df.dropna(subset=[1])
#         vals = df.values
#     except:
#         print("VCF 读取回退到标准模式...")
#         df = pd.read_csv(vcf_path, sep=r"\s+", dtype={"chr": str})
#         vals = df.values
#
#     records = {"normal": [], "mutation": []}
#
#     for row in vals:
#         chrom, pos, ref, alt = row[0], int(row[1]), row[2], row[3]
#         ref_seq, mut_seq = extract_sequence(chrom, pos, ref, alt)
#         if ref_seq is None: continue
#
#         seq_id = f"{chrom}_{pos}_{ref}to{alt}"
#         records["normal"].append(SeqRecord(Seq(ref_seq), id=f"{seq_id}_normal", description=""))
#         records["mutation"].append(SeqRecord(Seq(mut_seq), id=f"{seq_id}_mutation", description=""))
#
#     print(f"序列生成完成: {len(records['normal'])} 条")
#
#     # 1. 保存 FASTA
#     for st in ["normal", "mutation"]:
#         SeqIO.write(records[st], os.path.join(out_dir, f"{name}_{st}.fasta"), "fasta")
#
#     # 2. 提取特征
#     for seq_type in ["normal", "mutation"]:
#         fasta_file = os.path.join(out_dir, f"{name}_{seq_type}.fasta")
#
#         # (A) 标准特征
#         for feat in features:
#             author_name = get_author_style_name(feat)
#             out_file = os.path.join(out_dir, f"{author_name}_{seq_type}.txt")
#             run_ifeature_wrapper(fasta_file, feat, out_file)
#
#         # (B) 单独计算 RCKmer
#         calculate_manual_RCKmer(fasta_file, out_dir, seq_type)
#
#     # 3. 计算 Diff
#     print(f"\n正在为 {name} 计算 Diff 特征...")
#
#     # 包含 RCKmer 的所有特征名
#     all_target_features = [get_author_style_name(f) for f in features] + ["featureRC2mer"]
#
#     for safe_name in all_target_features:
#         norm_file = os.path.join(out_dir, f"{safe_name}_normal.txt")
#         mut_file = os.path.join(out_dir, f"{safe_name}_mutation.txt")
#         # 输出文件名改为作者格式: diffe_feature...
#         diff_file = os.path.join(out_dir, f"diffe_{safe_name}.txt")
#
#         if not (os.path.exists(norm_file) and os.path.exists(mut_file)):
#             continue
#
#         try:
#             df_n = pd.read_csv(norm_file)
#             df_m = pd.read_csv(mut_file)
#
#             # 仅数值列
#             cols = df_n.select_dtypes(include=['number']).columns
#             df_diff = df_m[cols] - df_n[cols]
#
#             # 如果原文件第一列是 ID (字符串)，可以插回去，也可以不插
#             # 作者文件貌似是纯数值，这里为了方便后续合并，建议只留数值
#             # 或者像你之前那样保留 ID，只要你的 DataLoader 能处理即可
#             # 这里为了安全，只保存数值 Diff
#
#             df_diff.to_csv(diff_file, sep="\t", index=False)
#             # print(f"  Success Diff → {os.path.basename(diff_file)}")
#
#         except Exception as e:
#             print(f"  Failed Diff {safe_name}: {e}")
#
#     print(f"\n{name} 全部处理完成！\n")
#
#
# if __name__ == "__main__":
#     for ds in datasets:
#         process_dataset(ds["name"], ds["vcf"], ds["out_dir"])
#     print("所有任务完成！特征已完全对齐作者参数 (11bp, k=2, RCKmer).")



# 之前物化特征生成
# """
# 终极修复版：解决 CKSNAP / MMI / Z_curve_9bit 永远返回空的问题
# 核心：import_parameters 后必须手动给这三个特征赋值参数，否则默认值无效！
# """
#
# import os
# import pandas as pd
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# import sys
#
# # ============ 路径配置 ============
# sys.path.append("/data/gyy/Project/Feature-Tool/iFeatureOmega-CLI")
# from iFeatureOmegaCLI import iDNA
#
# hg19_path = "/data/gyy/Data/hg19.fa/hg19.fa"
# param_file = "/data/gyy/Project/Feature-Tool/iFeatureOmega-CLI/parameters/DNA_parameters_setting.json"
#
# if not os.path.exists(hg19_path):
#     raise FileNotFoundError(f"参考基因组文件不存在: {hg19_path}")
#
# print("正在加载 hg19 基因组...")
# genome = SeqIO.to_dict(SeqIO.parse(hg19_path, "fasta"))
#
# FLANK = 52
# TOTAL_LEN = 2 * FLANK + 1  # 105bp，完美兼容所有特征（尤其是 Z_curve_9bit）
#
# features = [
#     "Kmer type 1", "RCKmer type 1", "NAC", "CKSNAP type 1",
#     "Mismatch", "MMI", "NMBroto",
#     "Z_curve_9bit", "Z_curve_12bit", "Z_curve_36bit",
#     "Z_curve_48bit", "Z_curve_144bit"
# ]
#
# # 把 datasets 改成 COSMIC
# datasets = [
#     {
#         "name": "COSMIC",
#         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/COSMIC_training.vcf",
#         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/train"
#     },
#     {
#         "name": "COSMIC",
#         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset/COSMIC_testing.vcf",
#         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/test"
#     }
# ]
# # datasets = [
# #     {
# #         "name": "SomaMutDB",
# #         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/SomaMutDBpart/SomaMutDB.vcf",
# #         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/SomaMutDB"
# #     },
# #     {
# #         "name": "EOSM",
# #         "vcf": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/EOSM/EOSM.vcf",
# #         "out_dir": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/physicochemical/EOSM"
# #     }
# # ]
#
# for ds in datasets:
#     os.makedirs(ds["out_dir"], exist_ok=True)
#
#
# def extract_sequence(chrom, pos, ref, alt):
#     key = str(chrom)
#     if key not in genome:
#         if f"chr{key}" in genome:
#             key = f"chr{key}"
#         elif key.startswith("chr") and key[3:] in genome:
#             key = key[3:]
#         else:
#             return None, None
#
#     seq = genome[key].seq
#     center = pos - 1
#     start = center - FLANK
#     end = center + FLANK + 1
#
#     # 边界处理 + 填充 A（比 N 更安全）
#     segment = seq[max(start, 0): min(end, len(seq))]
#     segment = str(segment).upper()
#
#     # 构造完整 105bp 序列
#     left = max(0, -start)
#     right = TOTAL_LEN - len(segment) - left
#
#     ref_seq = "A" * left + segment + "A" * right
#
#     # 强制清理非 ACGT 字符
#     ref_seq = ''.join(b if b in 'ACGT' else 'A' for b in ref_seq)
#
#     # 构建突变序列（注意处理插入/缺失，但你的数据都是 SNP，所以简单处理即可）
#     try:
#         mut_seq = ref_seq[:FLANK] + alt.upper() + ref_seq[FLANK + len(ref):]
#     except:
#         mut_seq = ref_seq[:FLANK] + alt.upper() + ref_seq[FLANK:]
#
#     # 保证长度一致
#     if len(mut_seq) > TOTAL_LEN:
#         mut_seq = mut_seq[:TOTAL_LEN]
#     elif len(mut_seq) < TOTAL_LEN:
#         mut_seq = mut_seq.ljust(TOTAL_LEN, 'A')
#
#     return ref_seq, mut_seq
#
#
# def extract_features_safe(fasta_file, name, seq_type, out_dir):
#     print(f"\n提取 {name}_{seq_type} 的特征...")
#
#     for feat in features:
#         safe_name = feat.replace(" ", "_").replace("(", "").replace(")", "")
#         out_file = os.path.join(out_dir, f"{name}_{seq_type}_{safe_name}.txt")
#         if os.path.exists(out_file):
#             os.remove(out_file)
#
#         dna = iDNA(fasta_file)
#
#         # 先尝试加载参数（就算失败也没事，后面会强制覆盖关键参数）
#         try:
#             dna.import_parameters(param_file)
#         except:
#             pass
#
#         # 【关键修复】对三个最容易寄的特征手动设置参数
#         if "CKSNAP" in feat:
#             dna.k = 6
#             dna.gap = 5
#             dna.normalize = True
#         elif feat == "MMI":
#             dna.nlag = 3          # 必须设置！否则返回空
#             dna.normalize = True
#         elif "Z_curve_9bit" in feat:
#             dna.zcurve = "9bit"   # 必须显式指定
#         elif "Z_curve_12bit" in feat:
#             dna.zcurve = "12bit"
#         elif "Z_curve_36bit" in feat:
#             dna.zcurve = "36bit"
#         elif "Z_curve_48bit" in feat:
#             dna.zcurve = "48bit"
#         elif "Z_curve_144bit" in feat:
#             dna.zcurve = "144bit"
#
#         try:
#             dna.get_descriptor(feat)
#             if hasattr(dna, 'encodings') and dna.encodings is not None and len(dna.encodings) > 0:
#                 dna.to_csv(out_file, index=False, header=True)
#                 print(f"  Success {feat.ljust(20)} → OK ({len(dna.encodings)} 条)")
#             else:
#                 print(f"  Failed {feat.ljust(20)} → 返回空")
#         except Exception as e:
#             print(f"  Failed {feat.ljust(20)} → 异常: {e}")
#
#
# def process_dataset(name, vcf_path, out_dir):
#     print(f"\n{'=' * 60}")
#     print(f"正在处理数据集: {name}  →  {vcf_path}")
#     print(f"{'=' * 60}")
#
#     # 1. 读取 VCF
#     df = pd.read_csv(vcf_path, sep=r"\s+", header=0,
#                      names=["chr", "pos", "ref", "alt"] if "EOSM" in vcf_path else None,
#                      dtype={"chr": str})
#     if "chr" not in df.columns:
#         df = pd.read_csv(vcf_path, sep=r"\s+", header=0, dtype={"chr": str})
#     df["pos"] = pd.to_numeric(df["pos"], errors='coerce').astype(int)
#
#     records = {"normal": [], "mutation": []}
#     skipped = 0
#
#     for _, row in df.iterrows():
#         chrom, pos, ref, alt = row["chr"], int(row["pos"]), str(row["ref"]), str(row["alt"])
#         ref_seq, mut_seq = extract_sequence(chrom, pos, ref, alt)
#         if ref_seq is None:
#             skipped += 1
#             continue
#
#         # 关键修复：ID 里绝对不能有 > 符号！
#         seq_id = f"{chrom}_{pos}_{ref}to{alt}"
#
#         records["normal"].append(SeqRecord(Seq(ref_seq),
#                                            id=f"{seq_id}_normal",
#                                            description=""))
#         records["mutation"].append(SeqRecord(Seq(mut_seq),
#                                              id=f"{seq_id}_mutation",
#                                              description=""))
#
#     print(f"序列生成完成，共 {len(records['normal'])} 条，跳过 {skipped} 条无法提取的位点")
#
#     for st in ["normal", "mutation"]:
#         fasta_path = os.path.join(out_dir, f"{name}_{st}.fasta")
#         SeqIO.write(records[st], fasta_path, "fasta")
#
#     # 2. 提取特征（使用安全版）
#     for seq_type in ["normal", "mutation"]:
#         fasta_file = os.path.join(out_dir, f"{name}_{seq_type}.fasta")
#         extract_features_safe(fasta_file, name, seq_type, out_dir)
#
#     # 3. 计算 Diff（超级稳健版，支持逗号分隔）
#     print(f"\n正在为 {name} 计算 Diff 特征...")
#     for feat in features:
#         safe_name = feat.replace(" ", "_").replace("(", "").replace(")", "")
#         norm_file = os.path.join(out_dir, f"{name}_normal_{safe_name}.txt")
#         mut_file  = os.path.join(out_dir, f"{name}_mutation_{safe_name}.txt")
#         diff_file = os.path.join(out_dir, f"{name}_diffe_{safe_name}.txt")
#
#         if not (os.path.exists(norm_file) and os.path.exists(mut_file)):
#             continue
#         if os.path.getsize(norm_file) < 100:
#             continue
#
#         try:
#             # 自动检测分隔符（逗号 or 制表符）
#             df_n = pd.read_csv(norm_file, sep=r'[,\t]', engine='python')
#             df_m = pd.read_csv(mut_file,  sep=r'[,\t]', engine='python')
#
#             # 自动识别 ID 列（通常是第一列，且包含序列ID）
#             id_col = df_n.columns[0]
#             feature_cols = df_n.columns[1:]
#
#             # 转为数值
#             df_n[feature_cols] = df_n[feature_cols].apply(pd.to_numeric, errors='coerce')
#             df_m[feature_cols] = df_m[feature_cols].apply(pd.to_numeric, errors='coerce')
#
#             # 计算差值
#             df_diff = df_m[feature_cols] - df_n[feature_cols]
#             df_diff.insert(0, id_col, df_n[id_col])  # 保留 ID 列
#
#             # 保存为制表符分隔（标准格式）
#             df_diff.to_csv(diff_file, sep="\t", index=False)
#             print(f"  Success Diff → {safe_name}.txt")
#
#         except Exception as e:
#             print(f"  Failed Diff {safe_name}: {e}")
#
#     print(f"\n{name} 所有特征提取 + Diff 计算完成！\n")
#
#
# if __name__ == "__main__":
#     for ds in datasets:
#         process_dataset(ds["name"], ds["vcf"], ds["out_dir"])
#
#     print("全部完成！SomaMutDB 和 EOSM 的 12 种物化特征（含 CKSNAP/MMI/Z_curve_9bit）已全部成功生成！")