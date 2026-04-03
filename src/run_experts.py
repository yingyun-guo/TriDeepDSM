import numpy as np
import os
import sys
from preprocess import process_and_save_tabular
import joblib

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from Datasets.Dataset_Integrated import IntegratedDataset
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ================= 配置路径 =================
SAVE_DIR = os.path.join(PROJECT_ROOT, "Datasets", "processed_meta")
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "out", "model")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

def get_top5_classifiers():
    return {
        # 1. HistGradientBoosting
        'HistGB': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42),
        # 2. LightGBM
        'LGBM': LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1,
                               random_state=42),
        # 3. XGBoost
        'XGB': XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, eval_metric='auc', n_jobs=-1,
                             random_state=42),
        # 4. RandomForest
        'RF': RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
        # 5. ExtraTrees
        'ERT': ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    }

def run_experts_cv():
    print(f"\n>>> [Phase 1] Loading Integrated Data <<<")
    # 1. 加载数据
    train_ds = IntegratedDataset('training')
    test_ds = IntegratedDataset('testing')

    X_train_raw = train_ds.feat_tabular
    y_train = train_ds.labels
    X_test_raw = test_ds.feat_tabular
    y_test = test_ds.labels

    print(f"   Raw Train Shape: {X_train_raw.shape}")
    print(f"   Raw Test Shape:  {X_test_raw.shape}")
    # ==========================================================================
    # 数据预处理 (QuantileTransformer)
    # ==========================================================================
    print(f"\n>>> [Phase 2] Preprocessing (QuantileTransformer) <<<")
    X_train, X_test = process_and_save_tabular(
        X_train_raw,
        X_test_raw,
        save_dir=MODEL_OUT_DIR,
        pipeline_name="tabular_pipeline.pkl"
    )

    # ==========================================================================
    # 训练 Experts
    # ==========================================================================
    classifiers = get_top5_classifiers()
    meta_train_list = []
    meta_test_list = []

    # 10 折交叉验证
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print(f"\n>>> [Phase 3] Training 5 Experts (CV Generation) <<<")
    for name, clf in classifiers.items():
        print(f"   Processing Expert: {name}...")
        # 1. 生成训练集 Meta特征
        prob_train = cross_val_predict(clf, X_train, y_train, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
        # 2. 生成测试集 Meta特征
        clf.fit(X_train, y_train)
        joblib.dump(clf, os.path.join(MODEL_OUT_DIR, f"{name}_expert.pkl"))
        prob_test = clf.predict_proba(X_test)[:, 1]

        # 3. 打印 AUC 监控性能
        auc_train = roc_auc_score(y_train, prob_train)
        auc_test = roc_auc_score(y_test, prob_test)
        print(f"-> Train CV AUC: {auc_train:.4f} | Test AUC: {auc_test:.4f}")

        # 收集结果
        meta_train_list.append(prob_train.reshape(-1, 1))
        meta_test_list.append(prob_test.reshape(-1, 1))
    # ==========================================================================
    # 保存结果
    # ==========================================================================
    print(f"\n>>> [Phase 4] Saving Meta-Features to {SAVE_DIR} <<<")

    meta_features_train = np.concatenate(meta_train_list, axis=1).astype(np.float32)
    meta_features_test = np.concatenate(meta_test_list, axis=1).astype(np.float32)

    np.save(os.path.join(SAVE_DIR, "meta_probs_5_training.npy"), meta_features_train)
    np.save(os.path.join(SAVE_DIR, "meta_probs_5_testing.npy"), meta_features_test)

    print(f"   Saved Train Meta Shape: {meta_features_train.shape}")
    print(f"   Saved Test Meta Shape:  {meta_features_test.shape}")
    print("   DONE!")


if __name__ == "__main__":
    run_experts_cv()


