import pandas as pd
import numpy as np
import os

# 自动获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_labels_from_vcf(vcf_path, force_label=None):
    if not os.path.exists(vcf_path):
        print(f" 文件不存在: {vcf_path}")
        return None

    print(f"正在读取文件: {vcf_path}")
    try:
        df = pd.read_csv(vcf_path, sep='\t')
        # 逻辑 1:  (针对 EOSM 和 SomaMutDB)
        if force_label is not None:
            count = len(df)
            labels = np.full(count, force_label, dtype=np.float32)
            print(f"  文件无 label 列，根据设置强制标记为: {force_label}")
        # 逻辑 2: 尝试从文件中读取 label 列
        else:
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
    base_data_dir = os.path.join(PROJECT_ROOT, "data")

    # ==========================================
    # 1. 有 Label 列的文件 (COSMIC 等)
    # ==========================================
    print("\n=== 加载含 Label 的文件 ===")
    y_train = load_labels_from_vcf(os.path.join(base_data_dir, "COSMIC_training.vcf"))
    y_test = load_labels_from_vcf(os.path.join(base_data_dir, "COSMIC_testing.vcf"))

    # ==========================================
    # 2. 无 Label 列的文件 (需手动指定)
    # ==========================================
    print("\n=== 加载无 Label 的文件 (手动指定) ===")
    y_eosm = load_labels_from_vcf(os.path.join(base_data_dir, "EOSM.vcf"), force_label=1)
    y_soma = load_labels_from_vcf(os.path.join(base_data_dir, "SomaMutDB.vcf"), force_label=0)

    print("\n=== 所有数据加载完成 ===")
