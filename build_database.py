import pandas as pd
import os

# ================= 配置路径 =================
DATA_DIR = "/data/gyy/Project/TriDeepDSM/data/"
MODEL_OUT_DIR = "/data/gyy/Project/TriDeepDSM/out/model/"
TRAIN_OUT_DIR = "/data/gyy/Project/TriDeepDSM/out/model/"

# 定义文件对应关系
DATASETS = [
    {
        "name": "COSMIC_Training",
        "vcf": os.path.join(DATA_DIR, "COSMIC_training.vcf"),
        "scores": os.path.join(TRAIN_OUT_DIR, "pred_train_scores.csv")
    },
    {
        "name": "COSMIC_Testing",
        "vcf": os.path.join(DATA_DIR, "COSMIC_testing.vcf"),
        "scores": os.path.join(MODEL_OUT_DIR, "pred_test_scores.csv")
    },
    {
        "name": "EOSM",
        "vcf": os.path.join(DATA_DIR, "EOSM.vcf"),
        "scores": os.path.join(MODEL_OUT_DIR, "pred_EOSM_scores.csv")
    },
    {
        "name": "SomaMutDB",
        "vcf": os.path.join(DATA_DIR, "SomaMutDB.vcf"),
        "scores": os.path.join(MODEL_OUT_DIR, "pred_SomaMutDB_scores.csv")
    }
]

all_dataframes = []

for ds in DATASETS:
    print(f"Processing {ds['name']}...")

    # 1. 读取 VCF
    # VCF 使用 \t 分隔
    df_vcf = pd.read_csv(ds['vcf'], sep='\t', low_memory=False)

    # 2. 读取 Scores
    df_scores = pd.read_csv(ds['scores'])

    # 3. 长度对齐 (防止特征提取时丢弃了最后几行)
    min_len = min(len(df_vcf), len(df_scores))
    df_vcf = df_vcf.iloc[:min_len].reset_index(drop=True)
    df_scores = df_scores.iloc[:min_len].reset_index(drop=True)

    # 4. 水平拼接
    # 我们保留 VCF 的所有列，再加上 scores 的 'prob' 列
    df_merged = pd.concat([df_vcf, df_scores[['prob']]], axis=1)

    # 5. 添加数据来源标记，方便网站后续展示
    df_merged['Dataset'] = ds['name']

    all_dataframes.append(df_merged)
    print(f"  -> Merged {min_len} rows.")

# ================= 合并并保存 =================
print("\n>>> Concatenating all datasets...")
final_database = pd.concat(all_dataframes, ignore_index=True)

# 计算最终的预测标签 (假设阈值是 0.5，你可以根据你的 optimal_params.json 修改)
THRESHOLD = 0.5
final_database['Prediction'] = final_database['prob'].apply(
    lambda x: 'Pathogenic (Driver)' if x > THRESHOLD else 'Benign (Passenger)'
)

# 统一重命名列名，让其更规范
final_database.rename(columns={'prob': 'TriDeepDSM_Score'}, inplace=True)

# 保存最终数据库
output_path = os.path.join(DATA_DIR, "TriDeepDSM_Master_Database.csv")
final_database.to_csv(output_path, index=False)

print("=" * 50)
print(f"SUCCESS! Master Database saved to {output_path}")
print(f"Total Mutations Available for Web Search: {len(final_database)}")
print(final_database.head())