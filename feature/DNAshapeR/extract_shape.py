import os
import pandas as pd
import numpy as np

# ================= 配置区域 =================
# 1. 你的 DNAshapeR 特征根目录
ROOT_SHAPE_TXT = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR"

# 2. 输出保存目录 (精确对齐刚才报错里提示的路径)
SAVE_DIR = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR/Datasets/processed_shape"
os.makedirs(SAVE_DIR, exist_ok=True)

# 3. 待处理的新数据集 (TP53, KRAS, LUSC)
DATASETS = ['TP53', 'KRAS', 'LUSC']

# 4. 特征类型与状态 (固定顺序，不可更改)
SHAPE_TYPES = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']
STATES = ['normal', 'mutation', 'diffe']

# ===========================================

def process_mode(mode):
    print(f"\n>>> 正在处理 {mode} 形状特征...")
    all_shapes = []

    for s_type in SHAPE_TYPES:
        for state in STATES:
            # 拼接文件名，例如: diffe_EP_TP53.txt
            fname = f"{state}_{s_type}_{mode}.txt"
            # 拼接文件路径，直接去对应名字的文件夹下找
            fpath = os.path.join(ROOT_SHAPE_TXT, mode, fname)

            if not os.path.exists(fpath):
                print(f"  [严重错误] 找不到文件: {fpath}")
                return

            try:
                # 读取 txt 数据
                df = pd.read_csv(fpath, sep=r'\s+', engine='python')
                df_num = df.select_dtypes([np.number])
                vals = df_num.values  # (N, L)

                # ================= 核心修复: 维度对齐 =================
                # HelT 和 Roll 是步长特征，通常只有 100 列，我们需要补齐到 101 列
                current_len = vals.shape[1]
                if current_len == 100:
                    vals = np.pad(vals, ((0, 0), (0, 1)), 'constant')
                # ====================================================

                # 扩展维度为 (N, 1, 101)
                vals = vals[:, np.newaxis, :]
                all_shapes.append(vals)
            except Exception as e:
                print(f"  [错误] 读取 {fname} 时发生异常: {e}")

    if not all_shapes:
        print("  [警告] 没有读取到任何数据！")
        return

    # 拼接所有的通道 (5 types * 3 states = 15 channels)
    try:
        shape_final = np.concatenate(all_shapes, axis=1)
        save_path = os.path.join(SAVE_DIR, f"shape_all_{mode}.npy")
        np.save(save_path, shape_final)
        print(f"  [完美成功] 已保存至: {save_path}")
        print(f"             最终维度: {shape_final.shape} (样本数, 15个通道, 101列)")
    except ValueError as e:
        print(f"  [拼接错误] {mode}: {e}")

if __name__ == "__main__":
    for m in DATASETS:
        process_mode(m)


# # gen_shape_npy.py
# import os
# import pandas as pd
# import numpy as np
#
# # ================= 配置区域 =================
# # 1. 你的 DNAshapeR 特征根目录
# ROOT_SHAPE_TXT = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR"
#
# # 2. 输出保存目录
# SAVE_DIR = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR/npy_files"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# # 3. 数据集配置 (Name -> {folder: 文件夹名, suffix: 文件后缀})
# # 根据你的 ls 结果设置
# DATASETS = {
#     "MFDSMC_train": {"folder": "MFDSMC", "suffix": "MFDSMC_train"},
#     "EPEL_train": {"folder": "EPEL", "suffix": "EPEL_train"}
# }
#
# # 4. 特征类型与状态
# SHAPE_TYPES = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']
# STATES = ['normal', 'mutation', 'diffe']
#
#
# # ===========================================
#
# def process_dataset(name, config):
#     print(f"\n>>> Processing {name}...")
#     folder = config['folder']
#     suffix = config['suffix']
#
#     all_shapes = []  # 用于存放 15 个通道的数据 (5 types * 3 states)
#
#     # 按照固定的顺序读取，确保通道顺序一致
#     # 顺序: EP(3states) -> HelT(3states) -> MGW(3states) -> ProT(3states) -> Roll(3states)
#     for s_type in SHAPE_TYPES:
#         for state in STATES:
#             # 构造文件名: e.g., diffe_EP_MFDSMC_test1.txt
#             fname = f"{state}_{s_type}_{suffix}.txt"
#             fpath = os.path.join(ROOT_SHAPE_TXT, folder, fname)
#
#             if not os.path.exists(fpath):
#                 print(f"   Error: Missing file: {fpath}")
#                 # 如果缺文件，整个数据集就无法对齐，必须停止
#                 return
#
#             try:
#                 # 读取空格分隔的 txt
#                 df = pd.read_csv(fpath, sep=r'\s+', engine='python')
#
#                 # 仅选择数值列 (有时候第一列可能是索引，防万一)
#                 df_num = df.select_dtypes([np.number])
#                 vals = df_num.values  # Shape: (N, L)
#
#                 # ================= 核心修复: 维度对齐 =================
#                 # MGW, ProT, EP 通常是 L=101
#                 # HelT, Roll 是步长特征，通常是 L=100
#                 # 我们统一 pad 到 101
#                 current_len = vals.shape[1]
#                 if current_len == 100:
#                     # 在最后一个维度(列) 补1列0 -> (N, 101)
#                     vals = np.pad(vals, ((0, 0), (0, 1)), 'constant')
#                 elif current_len != 101:
#                     print(f"  Warning: Unexpected length {current_len} in {fname}")
#                 # ====================================================
#
#                 # 扩展维度为 (N, 1, 101) 以便后续拼接 channel
#                 vals = vals[:, np.newaxis, :]
#                 all_shapes.append(vals)
#
#             except Exception as e:
#                 print(f"  Error reading {fname}: {e}")
#                 return
#
#     # 拼接所有通道
#     if all_shapes:
#         try:
#             # 沿 axis 1 拼接 -> (N, 15, 101)
#             shape_final = np.concatenate(all_shapes, axis=1)
#
#             save_name = f"{name}_shape.npy"
#             save_path = os.path.join(SAVE_DIR, save_name)
#             np.save(save_path, shape_final)
#
#             print(f"   Saved: {save_path}")
#             print(f"     Shape: {shape_final.shape} (Sample, Channel, Length)")
#
#         except ValueError as e:
#             print(f"  Error concatenating {name}: {e}")
#             # 打印详细形状以供调试
#             for i, arr in enumerate(all_shapes):
#                 print(f"    Channel {i} shape: {arr.shape}")
#
#
# if __name__ == "__main__":
#     for name, config in DATASETS.items():
#         process_dataset(name, config)



# import os
# import pandas as pd
# import numpy as np
#
# # ================= 配置区域 =================
# # 1. 你的 DNAshapeR 特征根目录
# ROOT_SHAPE_TXT = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR"
#
# # 2. 输出保存目录
# SAVE_DIR = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR/npy_files"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# # 3. 数据集配置 (Name -> {folder: 文件夹名, suffix: 文件后缀})
# # 根据你的 ls 结果设置
# DATASETS = {
#     "MFDSMC_test1": {"folder": "MFDSMC", "suffix": "MFDSMC_test1"},
#     "MFDSMC_test2": {"folder": "MFDSMC", "suffix": "MFDSMC_test2"},
#     "EPEL_test": {"folder": "EPEL", "suffix": "EPEL_test"}
# }
#
# # 4. 特征类型与状态
# SHAPE_TYPES = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']
# STATES = ['normal', 'mutation', 'diffe']
#
#
# # ===========================================
#
# def process_dataset(name, config):
#     print(f"\n>>> Processing {name}...")
#     folder = config['folder']
#     suffix = config['suffix']
#
#     all_shapes = []  # 用于存放 15 个通道的数据 (5 types * 3 states)
#
#     # 按照固定的顺序读取，确保通道顺序一致
#     # 顺序: EP(3states) -> HelT(3states) -> MGW(3states) -> ProT(3states) -> Roll(3states)
#     for s_type in SHAPE_TYPES:
#         for state in STATES:
#             # 构造文件名: e.g., diffe_EP_MFDSMC_test1.txt
#             fname = f"{state}_{s_type}_{suffix}.txt"
#             fpath = os.path.join(ROOT_SHAPE_TXT, folder, fname)
#
#             if not os.path.exists(fpath):
#                 print(f"   Error: Missing file: {fpath}")
#                 # 如果缺文件，整个数据集就无法对齐，必须停止
#                 return
#
#             try:
#                 # 读取空格分隔的 txt
#                 df = pd.read_csv(fpath, sep=r'\s+', engine='python')
#
#                 # 仅选择数值列 (有时候第一列可能是索引，防万一)
#                 df_num = df.select_dtypes([np.number])
#                 vals = df_num.values  # Shape: (N, L)
#
#                 # ================= 核心修复: 维度对齐 =================
#                 # MGW, ProT, EP 通常是 L=101
#                 # HelT, Roll 是步长特征，通常是 L=100
#                 # 我们统一 pad 到 101
#                 current_len = vals.shape[1]
#                 if current_len == 100:
#                     # 在最后一个维度(列) 补1列0 -> (N, 101)
#                     vals = np.pad(vals, ((0, 0), (0, 1)), 'constant')
#                 elif current_len != 101:
#                     print(f"  Warning: Unexpected length {current_len} in {fname}")
#                 # ====================================================
#
#                 # 扩展维度为 (N, 1, 101) 以便后续拼接 channel
#                 vals = vals[:, np.newaxis, :]
#                 all_shapes.append(vals)
#
#             except Exception as e:
#                 print(f"  Error reading {fname}: {e}")
#                 return
#
#     # 拼接所有通道
#     if all_shapes:
#         try:
#             # 沿 axis 1 拼接 -> (N, 15, 101)
#             shape_final = np.concatenate(all_shapes, axis=1)
#
#             save_name = f"{name}_shape.npy"
#             save_path = os.path.join(SAVE_DIR, save_name)
#             np.save(save_path, shape_final)
#
#             print(f"   Saved: {save_path}")
#             print(f"     Shape: {shape_final.shape} (Sample, Channel, Length)")
#
#         except ValueError as e:
#             print(f"  Error concatenating {name}: {e}")
#             # 打印详细形状以供调试
#             for i, arr in enumerate(all_shapes):
#                 print(f"    Channel {i} shape: {arr.shape}")
#
#
# if __name__ == "__main__":
#     for name, config in DATASETS.items():
#         process_dataset(name, config)


# # gen_shape_npy.py
# import os
# import pandas as pd
# import numpy as np
#
# # 配置路径
# ROOT_SHAPE_TXT = "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAshapeR"
# SAVE_DIR = "Datasets/processed_shape"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# SHAPE_TYPES = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']
# STATES = ['normal', 'mutation', 'diffe']
#
#
# def process_mode(mode):
#     print(f">>> Processing {mode}...")
#
#     # 确定文件夹名
#     folder_name = "COSMIC" if mode in ['training', 'testing'] else mode
#     # 确定后缀
#     suffix = "training" if mode == 'training' else "testing"
#     if mode not in ['training', 'testing']:
#         suffix = mode
#
#     all_shapes = []
#
#     for s_type in SHAPE_TYPES:
#         for state in STATES:
#             fname = f"{state}_{s_type}_{suffix}.txt"
#             fpath = os.path.join(ROOT_SHAPE_TXT, folder_name, fname)
#
#             if not os.path.exists(fpath):
#                 print(f"  [Warn] Missing: {fname} -> Skipping (Risk of misalignment!)")
#                 continue
#
#             try:
#                 # 读取
#                 df = pd.read_csv(fpath, sep=r'\s+', engine='python')
#                 df_num = df.select_dtypes([np.number])
#                 vals = df_num.values  # (N, L)
#
#                 # ================= 核心修复 =================
#                 # 检测维度: HelT/Roll 只有 100，其他有 101
#                 # 统一补齐到 101
#                 current_len = vals.shape[1]
#                 if current_len == 100:
#                     # 在最后一个维度补1列0 -> (N, 101)
#                     vals = np.pad(vals, ((0, 0), (0, 1)), 'constant')
#                 # ===========================================
#
#                 # 扩展维度为 (N, 1, 101)
#                 vals = vals[:, np.newaxis, :]
#                 all_shapes.append(vals)
#             except Exception as e:
#                 print(f"  Error reading {fname}: {e}")
#
#     if not all_shapes:
#         print("  No data found!")
#         return
#
#     # 拼接
#     try:
#         shape_final = np.concatenate(all_shapes, axis=1)
#         print(f"  Done. Shape: {shape_final.shape}")
#
#         save_path = os.path.join(SAVE_DIR, f"shape_all_{mode}.npy")
#         np.save(save_path, shape_final)
#         print(f"  Saved to {save_path}")
#     except ValueError as e:
#         print(f"  Error concatenating {mode}: {e}")
#         # 打印详细形状以供调试
#         for i, arr in enumerate(all_shapes):
#             print(f"    Index {i}: {arr.shape}")
#
#
# # 运行处理
# # 注意列表里每个元素都是字符串
# for m in ['training', 'testing', 'EOSM', 'SomaMutDB']:
#     process_mode(m)