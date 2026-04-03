import os
import sys
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
import joblib
import pandas as pd
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from Datasets.dataset_triview import TriViewDataset
from Datasets.dataset_tabular import IntegratedDataset
from models.TriView_Net import TriView_Net
import warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# 高级特征选择器 (融合了 去冗余 + 树模型排名 + 递归消除)
# ==============================================================================
class AdvancedFeatureSelector:
    def __init__(self, output_dir, correlation_threshold=0.95, step=5, min_features_to_select=30):
        self.output_dir = output_dir
        self.corr_thresh = correlation_threshold
        self.step = step
        self.min_features = min_features_to_select

        # 保存最终保留的列索引
        self.selected_indices_ = None
        # 保存列名（如果有）
        self.selected_columns_ = None

    def fit(self, X, y, feature_names=None):
        """
        X: numpy array or dataframe
        y: labels
        """
        print("\n>>> [Advanced Feature Selection] Start...")
        y = np.array(y).astype(int)
        if isinstance(X, pd.DataFrame):
            X_vals = X.values
            feat_names = X.columns.tolist()
        else:
            X_vals = X
            feat_names = [f"feat_{i}" for i in range(X.shape[1])] if feature_names is None else feature_names

        # === Step 1: 移除低方差特征 (Cleaning) ===
        print("   Phase 1: Removing low variance features...")
        var_selector = VarianceThreshold(threshold=0)
        X_var = var_selector.fit_transform(X_vals)
        remain_idx_1 = np.where(var_selector.get_support())[0]
        print(f"      Remaining: {len(remain_idx_1)} / {X_vals.shape[1]}")

        # === Step 2: 初步筛选 (Importance) ===
        print("   Phase 2: Pre-filtering with XGBoost (Importance)...")

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            n_jobs=-1,
            eval_metric='logloss',
            objective='binary:logistic',
            random_state=42
        )
        xgb.fit(X_var, y)

        # 获取特征重要性
        importances = xgb.feature_importances_
        # 保留 Top 400 进入下一轮
        top_k = min(400, X_var.shape[1])
        top_k_idx = np.argsort(importances)[::-1][:top_k]

        # 映射回原始索引
        current_indices = remain_idx_1[top_k_idx]
        X_curr = X_vals[:, current_indices]
        print(f"      Top {top_k} features selected by Importance.")

        # === Step 3: 相关性去冗余 (Pearson Correlation) ===
        print(f"   Phase 3: Removing highly correlated features (Threshold > {self.corr_thresh})...")
        df_temp = pd.DataFrame(X_curr)
        corr_matrix = df_temp.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_local_idx = [column for column in upper.columns if any(upper[column] > self.corr_thresh)]

        keep_local_idx = [i for i in range(X_curr.shape[1]) if i not in to_drop_local_idx]
        final_indices_pool = current_indices[keep_local_idx]
        X_final_pool = X_vals[:, final_indices_pool]
        print(f"      Dropped {len(to_drop_local_idx)} features. Remaining pool: {len(final_indices_pool)}")

        # === Step 4: RFECV (递归特征消除) ===
        print("   Phase 4: Running RFECV (Wrapper Method) to find optimal subset...")
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=100,
            n_jobs=-1,  # 并行加速
            verbosity=-1,  # 静默模式
            random_state=42
        )

        rfecv = RFECV(
            estimator=clf,
            step=self.step,
            cv=StratifiedKFold(5),
            scoring='roc_auc',
            min_features_to_select=self.min_features,
            n_jobs=-1
        )

        rfecv.fit(X_final_pool, y)

        # 最终选择的索引
        best_local_indices = np.where(rfecv.support_)[0]
        self.selected_indices_ = final_indices_pool[best_local_indices]

        print(f"   >>> DONE. Optimal number of features: {rfecv.n_features_}")
        print(f"       Best CV Score (AUC): {max(rfecv.cv_results_['mean_test_score']):.4f}")

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_].values
        return X[:, self.selected_indices_]


# ==============================================================================
# 日志与指标工具
# ==============================================================================
def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file = os.path.join(output_dir, "training_lightgbm_history.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.0
    precision_prc, recall_prc, _ = metrics.precision_recall_curve(y_true, y_prob)
    prc_auc = metrics.auc(recall_prc, precision_prc)
    return {
        "ACC": metrics.accuracy_score(y_true, y_pred),
        "MCC": metrics.matthews_corrcoef(y_true, y_pred),
        "Recall": metrics.recall_score(y_true, y_pred),
        "Precision": metrics.precision_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "F1": metrics.f1_score(y_true, y_pred),
        "AUC": roc_auc,
        "AUPR": prc_auc,
        "Matrix": f"[TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}]"
    }


def log_metrics(logger, title, res):
    logger.info(f"--- {title} ---")
    logger.info(f"   AUC : {res['AUC']:.4f} | AUPR: {res['AUPR']:.4f}")
    logger.info(f"   ACC : {res['ACC']:.4f} | F1  : {res['F1']:.4f}")
    logger.info(f"   MCC : {res['MCC']:.4f} | Spec: {res['Specificity']:.4f} | Sens: {res['Recall']:.4f}")


# ==============================================================================
# 特征提取类 (Deep Feature Extraction)
# ==============================================================================
class DeepFeatureExtractor:
    def __init__(self, config, device, logger):
        self.config = config
        self.device = device
        self.logger = logger
        # 修正：维度应该是 256(BERT) + 128(Shape) + 64(Meta) = 448
        self.feature_dim = 448

    def _load_single_model(self, fold):
        model_dir = self.config["PRETRAINED_MODEL_DIR"]
        model_path = os.path.join(model_dir, f"TriView_Fold_{fold + 1}.pth")
        if not os.path.exists(model_path):
            self.logger.warning(f"Model file not found: {model_path}")
            return None
        model = TriView_Net(config=self.config["MODEL_PARAMS"]).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def extract_oof(self, dataset):
        import gc
        num_samples = len(dataset)
        # 修正：维度改为 self.feature_dim (448)
        oof_features = np.zeros((num_samples, self.feature_dim), dtype=np.float32)

        indices_path = os.path.join(self.config["PRETRAINED_MODEL_DIR"], "kfold_indices.pkl")
        self.logger.info(f">>> Loading K-Fold Indices from {indices_path}...")
        folds_indices = joblib.load(indices_path)

        self.logger.info(">>> Starting Deep OOF Feature Extraction...")
        for fold, (train_idx, val_idx) in enumerate(folds_indices):
            model = self._load_single_model(fold)
            if model is None: continue

            val_sub = Subset(dataset, val_idx)
            val_loader = DataLoader(val_sub, batch_size=self.config["BATCH_SIZE"], shuffle=False, num_workers=2)

            fold_preds = []
            with torch.no_grad():
                for spatial, bert, meta, label in val_loader:
                    # 修正1：meta 放入 device
                    spatial, bert, meta = spatial.to(self.device), bert.to(self.device), meta.to(self.device)

                    f_hdr = model.hdr_branch(bert)
                    f_stf = model.stf_branch(spatial)
                    # 修正2：提取 meta 特征
                    f_meta = model.meta_fc(meta)

                    # 修正3：加入拼接
                    f_combined = torch.cat([f_hdr, f_stf, f_meta], dim=1)
                    fold_preds.append(f_combined.cpu().numpy())

            if len(fold_preds) > 0:
                oof_features[val_idx] = np.concatenate(fold_preds, axis=0)

            if (fold + 1) % 2 == 0:
                self.logger.info(f"   Processed OOF Fold {fold + 1}/{self.config['N_FOLDS']}")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        return oof_features

    def extract_test_ensemble(self, dataloader):
        import gc
        total_samples = len(dataloader.dataset)
        accumulated_features = np.zeros((total_samples, self.feature_dim), dtype=np.float32)
        models_loaded_count = 0

        self.logger.info(">>> Starting Test Ensemble Feature Extraction...")
        for fold in range(self.config["N_FOLDS"]):
            model = self._load_single_model(fold)
            if model is None: continue

            fold_feats = []  # <--- 这里定义的是 fold_feats
            with torch.no_grad():
                for spatial, bert, meta, label in dataloader:
                    spatial, bert, meta = spatial.to(self.device), bert.to(self.device), meta.to(self.device)

                    f_hdr = model.hdr_branch(bert)
                    f_stf = model.stf_branch(spatial)
                    f_meta = model.meta_fc(meta)

                    f_combined = torch.cat([f_hdr, f_stf, f_meta], dim=1)

                    # === 修改这里 ===
                    # 原代码（报错）：fold_preds.append(f_combined.cpu().numpy())
                    fold_feats.append(f_combined.cpu().numpy())
                    # ===============

            # 下面引用的也是 fold_feats，所以必须保持一致
            if len(fold_feats) > 0:
                current_fold_data = np.concatenate(fold_feats, axis=0)
                if len(current_fold_data) == total_samples:
                    accumulated_features += current_fold_data
                    models_loaded_count += 1

            del model
            gc.collect()
            torch.cuda.empty_cache()

        if models_loaded_count == 0: return np.zeros((total_samples, self.feature_dim))
        return accumulated_features / models_loaded_count


# ==============================================================================
# SECTION 3: 主程序
# ==============================================================================
if __name__ == "__main__":
    CONFIG = {
        "PATH_META": os.path.join(PROJECT_ROOT, "Datasets", "processed_meta"),
        "PATH_SHAPE": os.path.join(PROJECT_ROOT, "Datasets", "processed_shape"),
        "PATH_BERT": os.path.join(PROJECT_ROOT, "feature", "DNAbert-2"),
        "PATH_VCF_DIR": os.path.join(PROJECT_ROOT, "data"),
        "PRETRAINED_MODEL_DIR": os.path.join(PROJECT_ROOT, "out", "model"),
        "OUTPUT_DIR": os.path.join(PROJECT_ROOT, "out", "model"),
        "MODEL_PARAMS": {"bert_in_dim": 768, "spatial_in_dim": 19, "num_experts": 5, "fusion_dropout": 0.2},
        "N_FOLDS": 10, "BATCH_SIZE": 64, "GPU_ID": "0",
        # --- 特征选择配置 ---
        "FEATURE_SELECTION": True,
        "MIN_FEATURES": 200,
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["GPU_ID"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logging(CONFIG["OUTPUT_DIR"])
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")

    # 1. 加载数据
    logger.info(">>> Loading Datasets...")
    train_ds_tri = TriViewDataset(mode='training', config=CONFIG)
    test_ds_tri = TriViewDataset(mode='testing', config=CONFIG)
    train_ds_tab = IntegratedDataset('training')
    test_ds_tab = IntegratedDataset('testing')

    y_train = train_ds_tri.labels
    X_train_tab = train_ds_tab.feat_tabular
    y_test = test_ds_tri.labels
    X_test_tab = test_ds_tab.feat_tabular

    # 长度对齐
    min_len_train = min(len(y_train), len(X_train_tab))
    y_train = y_train[:min_len_train]
    X_train_tab = X_train_tab[:min_len_train]
    min_len_test = min(len(y_test), len(X_test_tab))
    y_test = y_test[:min_len_test]
    X_test_tab = X_test_tab[:min_len_test]

    # 2. Tabular 数据预处理 (Fit on Train, Transform Test)
    logger.info(">>> Preprocessing Tabular Features...")
    tabular_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))
    ])
    X_train_tab = tabular_pipeline.fit_transform(X_train_tab)
    X_test_tab = tabular_pipeline.transform(X_test_tab)

    # 保存 Pipeline
    joblib.dump(tabular_pipeline, os.path.join(CONFIG["OUTPUT_DIR"], "tabular_pipeline.pkl"))

    # 3. 提取 Deep Features
    extractor = DeepFeatureExtractor(CONFIG, DEVICE, logger)

    logger.info(">>> Extracting Train Deep Features (Layer 1 OOF)...")
    X_train_deep = extractor.extract_oof(train_ds_tri)[:min_len_train]

    logger.info(">>> Extracting Test Deep Features (Ensemble)...")
    test_loader = DataLoader(test_ds_tri, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2)
    X_test_deep = extractor.extract_test_ensemble(test_loader)[:min_len_test]

    # 4. 特征融合
    logger.info(">>> Fusing Features...")
    meta_train = train_ds_tri.feat_meta[:min_len_train]
    meta_test = test_ds_tri.feat_meta[:min_len_test]

    X_train_fused = np.concatenate([X_train_deep, X_train_tab, meta_train], axis=1)
    X_test_fused = np.concatenate([X_test_deep, X_test_tab, meta_test], axis=1)
    logger.info(f"Fused Shape: {X_train_fused.shape}")

    # 清洗 NaN/Inf
    logger.info(">>> Cleaning Fused Features (Handling NaN/Inf)...")
    X_train_fused = np.nan_to_num(X_train_fused, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X_test_fused = np.nan_to_num(X_test_fused, nan=np.nan, posinf=np.nan, neginf=np.nan)
    fusion_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train_fused = fusion_imputer.fit_transform(X_train_fused)
    X_test_fused = fusion_imputer.transform(X_test_fused)
    joblib.dump(fusion_imputer, os.path.join(CONFIG["OUTPUT_DIR"], "fusion_imputer.pkl"))

    # ==========================================================================
    # 5. 全局高级特征选择
    # ==========================================================================
    if CONFIG["FEATURE_SELECTION"]:
        logger.info(">>> Running Advanced Feature Selection (Global)...")

        # 初始化高级选择器
        adv_selector = AdvancedFeatureSelector(
            output_dir=CONFIG["OUTPUT_DIR"],
            correlation_threshold=0.98,  # 去除高度冗余
            step=20,  # 每次递归删除5个特征，加速
            min_features_to_select=CONFIG["MIN_FEATURES"]
        )
        # 训练选择器
        adv_selector.fit(X_train_fused, y_train)
        # 转换数据
        X_train_final = adv_selector.transform(X_train_fused)
        X_test_final = adv_selector.transform(X_test_fused)
        # 保存选择器
        joblib.dump(adv_selector, os.path.join(CONFIG["OUTPUT_DIR"], "final_feature_selector.pkl"))

        logger.info(f"Feature Selection Complete. New Shape: {X_train_final.shape}")
    else:
        logger.info(">>> Skipping Feature Selection.")
        X_train_final, X_test_final = X_train_fused, X_test_fused

    # ==========================================================================
    # 6. [GridSearchCV] 寻找最佳参数
    # ==========================================================================
    logger.info(">>> Running GridSearchCV...")
    # 修改 GridSearchCV 部分
    param_grid = {
        'learning_rate': [0.01, 0.015],  # 降低学习率
        'num_leaves': [15, 20],  # 减小叶子节点 (之前是 20, 31)，防止过拟合
        'max_depth': [4, 5],  # 限制深度 (之前是 5, 7)
        'subsample': [0.6, 0.7],  # 增加行采样随机性
        'colsample_bytree': [0.5, 0.6],  # 增加列采样随机性，每次只看一半特征
        'reg_alpha': [0.1, 1.0],  # L1 正则化 (关键！增加稀疏性)
        'reg_lambda': [0.1, 1.0],  # L2 正则化
    }

    # 增加 min_child_samples 防止在叶子上样本太少
    base_lgbm = LGBMClassifier(
        n_estimators=1000,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
        class_weight='balanced',
        min_child_samples=30  # 新增
    )

    base_lgbm = LGBMClassifier(n_estimators=1000, random_state=42, n_jobs=4, verbosity=-1, class_weight='balanced')
    # 使用已经筛选好的 X_train_final
    grid = GridSearchCV(estimator=base_lgbm, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1, n_jobs=4)
    grid.fit(X_train_final, y_train)

    best_params = grid.best_params_
    best_params['n_estimators'] = 5000  # 最终训练增加树的数量
    best_params['n_jobs'] = -1
    logger.info(f"Best Params: {best_params}")

    # ==========================================================================
    # 7. Strict 10-Fold CV (OOF Generation)
    # ==========================================================================
    logger.info("\n>>> executing Strict 10-Fold CV (OOF)...")
    oof_probs = np.zeros(len(y_train))
    cv_metrics_list = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_final, y_train)):
        # 直接切分已经筛选好的特征
        X_tr, X_val = X_train_final[train_idx], X_train_final[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        clf_cv = LGBMClassifier(**best_params)
        clf_cv.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(150, verbose=False)]
        )

        prob_val = clf_cv.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = prob_val

        res = calculate_metrics(y_val, prob_val)
        cv_metrics_list.append(res)
        if (fold + 1) % 2 == 0:
            log_metrics(logger, f"CV Fold {fold + 1}", res)

    avg_cv = {k: np.mean([m[k] for m in cv_metrics_list]) for k in ["AUC", "ACC", "F1", "MCC"]}
    logger.info(f"\n>>> 10-Fold CV Summary: AUC: {avg_cv['AUC']:.4f} | ACC: {avg_cv['ACC']:.4f}")

    # ==========================================================================
    # 8. 阈值优化
    # ==========================================================================
    precision, recall, thresholds = metrics.precision_recall_curve(y_train, oof_probs)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    logger.info(f">>> Best Threshold: {best_threshold:.4f} (OOF F1: {best_f1:.4f})")

    with open(os.path.join(CONFIG["OUTPUT_DIR"], "optimal_params.json"), "w") as f:
        json.dump({"threshold": float(best_threshold), "train_f1": float(best_f1)}, f)

    # ==========================================================================
    # 9. 最终全量训练
    # ==========================================================================
    logger.info("\n>>> Retraining Final Model on Full Data...")
    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_train_final, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    final_clf = LGBMClassifier(**best_params)
    final_clf.fit(
        X_tr_final, y_tr_final,
        eval_set=[(X_val_final, y_val_final)],
        eval_metric='auc',
        callbacks=[early_stopping(150), log_evaluation(1000)]
    )

    joblib.dump(final_clf, os.path.join(CONFIG["OUTPUT_DIR"], "final_lightgbm.pkl"))

    # ==========================================================================
    # 10. 独立测试集评估
    # ==========================================================================
    y_prob_test = final_clf.predict_proba(X_test_final)[:, 1]
    test_res = calculate_metrics(y_test, y_prob_test)
    log_metrics(logger, "Independent Test Results", test_res)

    # 保存结果
    pd.DataFrame({'label': y_train, 'prob': oof_probs}).to_csv(
        os.path.join(CONFIG["OUTPUT_DIR"], "pred_train_oof_scores.csv"), index=False)
    pd.DataFrame({'label': y_test, 'prob': y_prob_test}).to_csv(
        os.path.join(CONFIG["OUTPUT_DIR"], "pred_test_scores.csv"), index=False)

    logger.info(">>> All Finished.")
