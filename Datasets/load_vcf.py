# #load_vcf.py代码
import pandas as pd
import numpy as np
import os


def load_labels_from_vcf(vcf_path, force_label=None):
    """
    读取 VCF/TXT 文件并提取 label 列。

    参数:
    vcf_path: 文件路径
    force_label: (可选) 如果文件里没有 label 列，强制指定所有样本的标签。
                 例如: 正样本集传 1, 负样本集传 0。
    """
    if not os.path.exists(vcf_path):
        print(f" 文件不存在: {vcf_path}")
        return None

    print(f"正在读取文件: {vcf_path}")
    try:
        # 读取 tab 分隔文件
        df = pd.read_csv(vcf_path, sep='\t')

        # 逻辑 1: 如果强制指定了标签 (针对 EOSM 和 SomaMutDB)
        if force_label is not None:
            count = len(df)
            labels = np.full(count, force_label, dtype=np.float32)
            print(f"  文件无 label 列，根据设置强制标记为: {force_label}")

        # 逻辑 2: 尝试从文件中读取 label 列
        else:
            # 清理列名空格
            df.columns = df.columns.str.strip()

            if 'label' not in df.columns:
                raise ValueError(f"文件中找不到 'label' 列且未指定 force_label！现有列: {df.columns.tolist()}")

            labels = df['label'].values.astype(np.float32)

        # 打印统计信息
        pos_count = int(sum(labels))
        neg_count = len(labels) - pos_count
        print(f"  成功提取标签: {len(labels)} 条 (正样本: {pos_count}, 负样本: {neg_count})")
        return labels

    except Exception as e:
        print(f"  读取失败: {e}")
        return None


if __name__ == "__main__":
    # 路径配置
    base_feature_dir = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC"
    base_dataset_dir = "/data/gyy/Project/DeepSTF-new/DeepSTF/dataset"

    # ==========================================
    # 1. 有 Label 列的文件 (COSMIC, EPEL, MFDSMC)
    # ==========================================
    print("\n=== 加载含 Label 的文件 ===")

    # COSMIC
    y_train = load_labels_from_vcf(f"{base_feature_dir}/COSMIC_training.vcf")
    y_test = load_labels_from_vcf(f"{base_feature_dir}/COSMIC_testing.vcf")

    # EPEL & MFDSMC
    y_epel = load_labels_from_vcf(f"{base_dataset_dir}/EPEL_test.vcf")
    y_mfds1 = load_labels_from_vcf(f"{base_dataset_dir}/MFDSMC_test1.vcf")
    y_mfds2 = load_labels_from_vcf(f"{base_dataset_dir}/MFDSMC_test2.txt")

    # ==========================================
    # 2. 无 Label 列的文件 (需手动指定)
    # ==========================================
    print("\n=== 加载无 Label 的文件 (手动指定) ===")

    # EOSM (正样本，强制 label=1)
    y_eosm = load_labels_from_vcf(
        f"{base_dataset_dir}/EOSM.vcf",
        force_label=1
    )

    # SomaMutDB (负样本，强制 label=0)
    y_soma = load_labels_from_vcf(
        f"{base_dataset_dir}/SomaMutDB.vcf",
        force_label=0
    )

    print("\n=== 所有数据加载完成 ===")


# # #load_vcf.py代码
# import numpy as np
#
#
# def load_labels_from_vcf(file_path):
#     labels = []
#     count = 0
#     with open(file_path, 'r') as f:
#         # 读取并跳过表头
#         header = f.readline()
#         for line in f:
#             if not line.strip():
#                 continue
#             count += 1
#             parts = line.strip().split('\t')
#             # 如果存在第5列 (label列)
#             if len(parts) >= 5:
#                 labels.append(int(float(parts[4])))
#             else:
#                 # 如果没有 label 列 (如 Case Study 数据)，默认填充 0
#                 labels.append(0)
#
#     return np.array(labels, dtype=np.int64)


#之前的
# import pandas as pd
# import numpy as np
#
#
# def load_labels_from_vcf(vcf_path):
#     print(f"正在读取 VCF 文件: {vcf_path}")
#     try:
#         # VCF 通常是 tab 分隔，有表头
#         df = pd.read_csv(vcf_path, sep='\t')
#
#         # 检查是否有 label 列
#         if 'label' not in df.columns:
#             raise ValueError(f"VCF 文件中找不到 'label' 列！现有列: {df.columns}")
#
#         labels = df['label'].values.astype(np.float32)
#         print(f"  成功提取标签: {len(labels)} 条 (正样本: {int(sum(labels))})")
#         return labels
#
#     except Exception as e:
#         print(f"  读取 VCF 失败: {e}")
#         return None
#
#
# # 测试一下
# if __name__ == "__main__":
#     train_vcf = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC/COSMIC_training.vcf"
#     test_vcf = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC/COSMIC_testing.vcf"
#
#     y_train = load_labels_from_vcf(train_vcf)
#     y_test = load_labels_from_vcf(test_vcf)