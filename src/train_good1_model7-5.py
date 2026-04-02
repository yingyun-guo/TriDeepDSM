# train_good1_model7.py
import os
import sys
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import joblib

# 引入数据集和模型
# 确保你的 Datasets/Dataset_TriView.py 是支持 config 参数的那个版本
from Datasets.Dataset_TriView_new import TriViewDataset
from models.TriView_Net_1 import TriView_Net


# ==============================================================================
# SECTION 1: 辅助函数 (保持不变)
# ==============================================================================
def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file = os.path.join(output_dir, "training_log.txt")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def calculate_comprehensive_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    acc = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = (recall + specificity) / 2.0
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0
    precision_prc, recall_prc, _ = metrics.precision_recall_curve(y_true, y_prob)
    prc_auc = metrics.auc(recall_prc, precision_prc)
    results = {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "ACC": acc, "Precision": precision, "Recall": recall, "MCC": mcc,
        "Specificity": specificity, "BACC": bacc, "F1": f1_score,
        "AUC": roc_auc, "AUPR": prc_auc
    }
    return results


def log_metrics(logger, title, res):
    logger.info(f"--- {title} ---")
    logger.info(f"   AUC : {res['AUC']:.4f} | AUPR: {res['AUPR']:.4f}")
    logger.info(f"   ACC : {res['ACC']:.4f} | BACC: {res['BACC']:.4f}")
    logger.info(f"   Matrix: [TN:{res['TN']}, FP:{res['FP']}, FN:{res['FN']}, TP:{res['TP']}]")


# ==============================================================================
# SECTION 2: 主执行逻辑
# ==============================================================================
if __name__ == '__main__':

    CONFIG = {
        # --- 路径配置 (Model-7) ---
        "PATH_META": "/data/gyy/Project/DeepSTF-new/DeepSTF/Datasets/processed_meta/model-7-5/",
        "PATH_SHAPE": "Datasets/processed_shape",
        "PATH_BERT": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/DNAbert-2",
        "PATH_VCF_DIR": "/data/gyy/Project/DeepSTF-new/DeepSTF/feature/COSMIC",

        # --- 输出配置 (Model-7) ---
        "OUTPUT_DIR": "/data/gyy/Project/DeepSTF-new/DeepSTF/out/model-7-5-3/",

        # --- 训练超参数 ---
        "GPU_ID": "0",
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.0005,
        "WEIGHT_DECAY": 1e-3,
        "EPOCHS": 40,
        "N_FOLDS": 10,
        "NUM_WORKERS": 2,

        # --- 模型结构参数 ---
        "MODEL_PARAMS": {
            "bert_in_dim": 768,
            "spatial_in_dim": 19,  # 19维特征
            "num_experts": 5,
            "fusion_dropout": 0.2
        }
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["GPU_ID"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    OUTPUT_ROOT_DIR = CONFIG["OUTPUT_DIR"]
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)

    logger = setup_logging(OUTPUT_ROOT_DIR)
    logger.info(">>> Experiment Started: Training Model-7 (Original Logic)")
    logger.info(f"   Device: {DEVICE}")

    # --- 加载数据 ---
    logger.info(">>> Loading Datasets...")
    # 注意：这里需要你之前的 TriViewDataset 修改已生效（支持 config 参数）
    full_train_ds = TriViewDataset(mode='training', config=CONFIG)

    # 【安全措施】如果之前报过 device-assert 错误，通常是因为数据里有 NaN
    # 为了保险起见，这里做一个简单的 inplace 替换，不影响逻辑结构
    full_train_ds.feat_spatial = np.nan_to_num(full_train_ds.feat_spatial)
    full_train_ds.feat_meta = np.nan_to_num(full_train_ds.feat_meta)

    # K-Fold
    skf = StratifiedKFold(n_splits=CONFIG["N_FOLDS"], shuffle=True, random_state=42)
    y_all = full_train_ds.labels

    # 1. 生成索引
    logger.info(">>> Generating and Saving K-Fold Indices...")
    folds_indices = list(skf.split(np.zeros(len(y_all)), y_all))

    # 2. 保存索引
    indices_path = os.path.join(OUTPUT_ROOT_DIR, "kfold_indices.pkl")
    joblib.dump(folds_indices, indices_path)
    logger.info(f"     K-Fold indices saved to: {indices_path}")

    cv_metrics_list = []
    start_time = time.time()

    # --- 开始 10折训练 ---
    for fold, (train_idx, val_idx) in enumerate(folds_indices):
        logger.info(f"\n=== Fold {fold + 1}/{CONFIG['N_FOLDS']} ===")

        # 1. 划分数据
        train_sub = Subset(full_train_ds, train_idx)
        val_sub = Subset(full_train_ds, val_idx)

        train_loader = DataLoader(train_sub, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                                  num_workers=CONFIG["NUM_WORKERS"])
        val_loader = DataLoader(val_sub, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                                num_workers=CONFIG["NUM_WORKERS"])

        # 2. 初始化模型
        model = TriView_Net(config=CONFIG["MODEL_PARAMS"]).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])

        # 保持原始 BCELoss
        criterion = nn.BCELoss()

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader),
                                                  epochs=CONFIG["EPOCHS"])

        best_val_auc = 0.0
        best_model_path = os.path.join(OUTPUT_ROOT_DIR, f"TriView_Fold_{fold + 1}.pth")

        # 3. 训练 Epoch
        for epoch in range(CONFIG["EPOCHS"]):
            model.train()
            for spatial, bert, meta, label in train_loader:
                spatial, bert, meta, label = spatial.to(DEVICE), bert.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
                optimizer.zero_grad()
                output = model(spatial, bert, meta)

                # 防止极其罕见的数值不稳定导致 output 略微越界 (e.g. 1.0000001)
                # 这是一个对 BCELoss 的最小保护，不改变核心逻辑
                output = torch.clamp(output, 1e-7, 1 - 1e-7)

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # 验证
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for spatial, bert, meta, label in val_loader:
                    spatial, bert, meta, label = spatial.to(DEVICE), bert.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
                    output = model(spatial, bert, meta)
                    preds.extend(output.cpu().numpy())
                    truths.extend(label.cpu().numpy())

            val_auc = metrics.roc_auc_score(truths, preds)

            if (epoch + 1) % 5 == 0:
                print(f"   [Epoch {epoch + 1}] Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        logger.info(f"   Fold {fold + 1} Best Val AUC: {best_val_auc:.4f} -> Saved.")

        # 4. 记录最佳性能
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for spatial, bert, meta, label in val_loader:
                spatial, bert, meta, label = spatial.to(DEVICE), bert.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
                output = model(spatial, bert, meta)
                preds.extend(output.cpu().numpy())
                truths.extend(label.cpu().numpy())

        fold_res = calculate_comprehensive_metrics(np.array(truths), np.array(preds))
        cv_metrics_list.append(fold_res)
        log_metrics(logger, f"Fold {fold + 1} Final Metrics", fold_res)

    # --- 汇总 ---
    logger.info("\n" + "=" * 40)
    logger.info(">>> CV Summary (Models Saved) <<<")
    avg_metrics = {}
    for key in cv_metrics_list[0].keys():
        values = [m[key] for m in cv_metrics_list]
        avg_metrics[key] = f"{np.mean(values):.4f} (+/- {np.std(values):.4f})"
    logger.info(json.dumps(avg_metrics, indent=2))

    end_time = time.time()
    logger.info(f"\n>>> Total Time: {(end_time - start_time) / 60:.2f} minutes")