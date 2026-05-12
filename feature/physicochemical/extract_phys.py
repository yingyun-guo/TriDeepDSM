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


