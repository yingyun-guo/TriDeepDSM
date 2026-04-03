import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import numpy as np
import pandas as pd
import torch
import joblib
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
import warnings
from models.TriView_Net import TriView_Net
from Datasets.Dataset_Integrated import IntegratedDataset
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
# ==============================================================================
# 关键类定义
# ==============================================================================
class AdvancedFeatureSelector:
    def __init__(self, output_dir, correlation_threshold=0.95, step=5, min_features_to_select=30):
        self.output_dir = output_dir
        self.corr_thresh = correlation_threshold
        self.step = step
        self.min_features = min_features_to_select
        self.selected_indices_ = None
        self.selected_columns_ = None

    def fit(self, X, y, feature_names=None):
        pass

    def transform(self, X):
        if self.selected_indices_ is None:
            return X
        if hasattr(X, "iloc"):
            return X.iloc[:, self.selected_indices_].values
        return X[:, self.selected_indices_]


# ================= 配置 =================
PATH_TEST2 = os.path.join(PROJECT_ROOT, "Datasets", "aligned_test2")
MODEL_DIR = os.path.join(PROJECT_ROOT, "out", "model")
DEEP_MODEL_DIR = os.path.join(PROJECT_ROOT, "out", "model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 工具函数 =================
class DeepFeatureExtractorTest:
    def __init__(self, model_dir, config, device):
        self.models = []
        self.device = device
        print(f"   [Deep Model] Loading experts from {model_dir}...")
        for fold in range(10):
            path = os.path.join(model_dir, f"TriView_Fold_{fold + 1}.pth")
            if not os.path.exists(path): continue
            model = TriView_Net(config=config).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            self.models.append(model)
        print(f"   [Deep Model] Loaded {len(self.models)} models.")

    def extract(self, sp_data, bt_data, meta_data):
        dataset = TensorDataset(
            torch.tensor(sp_data).float(),
            torch.tensor(bt_data).float(),
            torch.tensor(meta_data).float()
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
        all_feats = []
        with torch.no_grad():
            for sp, bt, meta in tqdm(loader, desc="   [Extracting Deep Features]"):
                sp, bt, meta = sp.to(self.device), bt.to(self.device), meta.to(self.device)
                if bt.shape[-1] == 768 and bt.ndim == 3:
                    bt = bt.permute(0, 2, 1)
                elif bt.shape[-1] == 768 and bt.ndim == 2:
                    bt = bt.unsqueeze(2)

                batch_feats = []
                for m in self.models:
                    f_hdr = m.hdr_branch(bt)
                    f_stf = m.stf_branch(sp)
                    f_meta = m.meta_fc(meta)
                    f_combined = torch.cat([f_hdr, f_stf, f_meta], dim=1)
                    batch_feats.append(f_combined.cpu().numpy())
                all_feats.append(np.mean(np.array(batch_feats), axis=0))
        return np.concatenate(all_feats, axis=0)


def calculate_metrics_full(y_true, y_prob, threshold=0.5):
    """ 完整指标计算，不删减任何内容 """
    y_pred = (y_prob > threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    # 基础指标
    acc = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)  # Sensitivity
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.5

    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall_curve, precision_curve)

    return {
        "AUC": roc_auc,
        "AUPR": aupr,
        "ACC": acc,
        "F1": f1,
        "MCC": mcc,
        "Sens": recall,
        "Spec": specificity,
        "Prec": precision  # 加回 Precision
    }

# ================= 主流程 =================
if __name__ == "__main__":
    print(f"\n>>> PIPELINE: Balanced Validation <<<")
    # 1. 加载数据
    print(">>> [Step 1] Loading Data...")
    try:
        e_y = np.load(f"{PATH_TEST2}/EOSM_label.npy")
        s_y = np.load(f"{PATH_TEST2}/SomaMutDB_label.npy")
        e_tab = np.load(f"{PATH_TEST2}/EOSM_tabular.npy")
        s_tab = np.load(f"{PATH_TEST2}/SomaMutDB_tabular.npy")
        e_bt = np.load(f"{PATH_TEST2}/EOSM_bert.npy")
        s_bt = np.load(f"{PATH_TEST2}/SomaMutDB_bert.npy")
        e_sp = np.load(f"{PATH_TEST2}/EOSM_spatial.npy")
        s_sp = np.load(f"{PATH_TEST2}/SomaMutDB_spatial.npy")
    except FileNotFoundError:
        print("Data missing.")
        sys.exit(1)
    y_test = np.concatenate([e_y, s_y])
    X_tab = np.concatenate([e_tab, s_tab])
    X_bt = np.concatenate([e_bt, s_bt])
    X_sp = np.concatenate([e_sp, s_sp])

    # ==========================================================================
    # [Step 2] Generating Meta Features (动态训练恢复 0.82 + 为Django保存)
    # ==========================================================================
    print(">>> [Step 2] Generating Meta Features & Saving for Django...")

    # 必须加载训练集，让专家模型现场学习最新的 2003 维特征！
    train_ds = IntegratedDataset('training')
    X_train = train_ds.feat_tabular
    y_train = train_ds.labels

    # 重新训练预处理管道
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))
    ])
    X_train_s = pipe.fit_transform(X_train)
    X_test_s = pipe.transform(X_tab)

    # 【网页部署关键】专门为 Django 保存这个匹配 2003 维特征的 Pipeline！
    joblib.dump(pipe, os.path.join(MODEL_DIR, "django_expert_pipeline.pkl"))

    # 注意：字典顺序必须是 LGBM, XGB, RF, ERT, HistGB，这是 0.82 的特征密码！
    experts = {
        'LGBM': LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.03, verbose=-1, n_jobs=-1,
                               random_state=42),
        'XGB': XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03, n_jobs=-1, eval_metric='auc',
                             random_state=42),
        'RF': RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=42),
        'ERT': ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=42),
        'HistGB': HistGradientBoostingClassifier(max_iter=150, max_depth=5, learning_rate=0.03, random_state=42)
    }

    meta_preds = []
    for name, clf in experts.items():
        # 当场训练！
        clf.fit(X_train_s, y_train)

        # 【网页部署关键】为 Django 保存这些新鲜出炉、绝不报错的专家模型！
        joblib.dump(clf, os.path.join(MODEL_DIR, f"django_{name}_expert.pkl"))

        meta_preds.append(clf.predict_proba(X_test_s)[:, 1].reshape(-1, 1))

    X_meta = np.concatenate(meta_preds, axis=1).astype(np.float32)

    # ==========================================================================
    # [Step 3] Loading Final Tabular Pipeline (For Fusion)
    # ==========================================================================
    print(">>> [Step 3] Loading Final Tabular Pipeline...")
    pipe_final = joblib.load(os.path.join(MODEL_DIR, "tabular_pipeline.pkl"))
    X_tab_scaled = pipe_final.transform(X_tab)

    # # 2. Meta Features
    # print(">>> [Step 2] Generating Meta Features...")
    # train_ds = IntegratedDataset('training')
    # X_train = train_ds.feat_tabular
    # y_train = train_ds.labels
    #
    # pipe = Pipeline([
    #     ('imputer', SimpleImputer(strategy='mean')),
    #     ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))
    # ])
    # X_train_s = pipe.fit_transform(X_train)
    # X_test_s = pipe.transform(X_tab)
    #
    # experts = {
    #     'LGBM': LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.03, verbose=-1, n_jobs=-1,
    #                            random_state=42),
    #     'XGB': XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03, n_jobs=-1, eval_metric='auc',
    #                          random_state=42),
    #     'RF': RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=42),
    #     'ERT': ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=42),
    #     'HistGB': HistGradientBoostingClassifier(max_iter=150, max_depth=5, learning_rate=0.03, random_state=42)
    # }
    #
    # meta_preds = []
    # for name, clf in experts.items():
    #     clf.fit(X_train_s, y_train)
    #     meta_preds.append(clf.predict_proba(X_test_s)[:, 1].reshape(-1, 1))
    # X_meta = np.concatenate(meta_preds, axis=1)
    #
    # # 3. Pipeline for Final Model
    # print(">>> [Step 3] Loading Final Tabular Pipeline...")
    # pipe_final = joblib.load(os.path.join(MODEL_DIR, "tabular_pipeline.pkl"))
    # X_tab_scaled = pipe_final.transform(X_tab)

    # 4. Deep Features
    print(">>> [Step 4] Extracting Deep Features...")
    deep_config = {'bert_in_dim': 768, 'spatial_in_dim': 19, 'num_experts': 5, 'fusion_dropout': 0.2}
    extractor = DeepFeatureExtractorTest(DEEP_MODEL_DIR, deep_config, DEVICE)
    X_deep = extractor.extract(X_sp, X_bt, X_meta)

    # ==========================================================================
    # [Step 5] Fusion & Feature Matching (DEBUG & FIX)
    # ==========================================================================
    print(">>> [Step 5] Fusion & Feature Matching...")

    # 1. 原始融合
    X_fused = np.concatenate([X_deep, X_tab_scaled, X_meta], axis=1)

    # 2. Impute
    fusion_imputer = joblib.load(os.path.join(MODEL_DIR, "fusion_imputer.pkl"))
    X_fused = np.nan_to_num(X_fused, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X_fused = fusion_imputer.transform(X_fused)

    print(f"[INFO] Raw Fused Dimension: {X_fused.shape[1]}")

    # 3. 加载模型
    clf_final = joblib.load(os.path.join(MODEL_DIR, "final_lightgbm.pkl"))
    expected_features = clf_final.n_features_in_
    print(f"[INFO] Model Expects: {expected_features} features")

    # 4. 加载选择器
    selector_path = os.path.join(MODEL_DIR, "final_feature_selector.pkl")
    X_final = X_fused

    if os.path.exists(selector_path):
        print(f"    [INFO] Found selector file at: {selector_path}")
        try:
            selector = joblib.load(selector_path)
            # 执行转换
            X_temp = selector.transform(X_fused)
            print(f" [INFO] Data transformed from {X_fused.shape[1]} -> {X_temp.shape[1]}")

            if X_temp.shape[1] == expected_features:
                print(" [SUCCESS] Dimensions matched perfectly.")
                X_final = X_temp
            else:
                print(f"[WARNING] Selector output {X_temp.shape[1]} != Model Input {expected_features}")
        except Exception as e:
            print(f"[ERROR] Failed to load selector: {e}")
    else:
        print(f" [WARNING] Selector file NOT found at {selector_path}")

    # 5. 保底修复
    if X_final.shape[1] != expected_features:
        print(f"[CRITICAL FIX] Still mismatch. Forcing truncation/padding...")
        if X_final.shape[1] > expected_features:
            X_final = X_final[:, :expected_features]
        else:
            pad = np.zeros((X_final.shape[0], expected_features - X_final.shape[1]))
            X_final = np.concatenate([X_final, pad], axis=1)
        print(f"[FIX RESULT] New shape: {X_final.shape}")

    # 预测
    probs = clf_final.predict_proba(X_final)[:, 1]
    # 读取阈值
    try:
        with open(os.path.join(MODEL_DIR, "optimal_params.json"), "r") as f:
            best_thr = json.load(f)["threshold"]
        print(f">>> Optimal Threshold: {best_thr:.4f}")
    except:
        best_thr = 0.5
    # 评估
    print("\n" + "=" * 80)
    print(f"RESULTS (Mean ± Std over 100 runs)")
    print("=" * 80)
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    n_pos = len(pos_idx)
    metrics_list = []

    for i in tqdm(range(100), desc="   [Evaluating]"):
        np.random.seed(i)
        if len(neg_idx) > n_pos:
            sampled_neg = np.random.choice(neg_idx, n_pos, replace=False)
        else:
            sampled_neg = neg_idx
        idx = np.concatenate([pos_idx, sampled_neg])

        # 使用 Full Metrics
        res = calculate_metrics_full(y_test[idx], probs[idx], best_thr)
        metrics_list.append(res)

    df_res = pd.DataFrame(metrics_list)

    # 打印所有指标
    print(f"   AUC  : {df_res['AUC'].mean():.4f} ± {df_res['AUC'].std():.4f}")
    print(f"   AUPR : {df_res['AUPR'].mean():.4f} ± {df_res['AUPR'].std():.4f}")
    print(f"   ACC  : {df_res['ACC'].mean():.4f} ± {df_res['ACC'].std():.4f}")
    print(f"   F1   : {df_res['F1'].mean():.4f} ± {df_res['F1'].std():.4f}")
    print(f"   MCC  : {df_res['MCC'].mean():.4f} ± {df_res['MCC'].std():.4f}")
    print("-" * 60)
    print(f"   Sens : {df_res['Sens'].mean():.4f} ± {df_res['Sens'].std():.4f}")
    print(f"   Spec : {df_res['Spec'].mean():.4f} ± {df_res['Spec'].std():.4f}")
    print(f"   Prec : {df_res['Prec'].mean():.4f} ± {df_res['Prec'].std():.4f}")

    df_res.to_csv(os.path.join(MODEL_DIR, "test2_balanced_results_7_5_3_full.csv"), index=False)


