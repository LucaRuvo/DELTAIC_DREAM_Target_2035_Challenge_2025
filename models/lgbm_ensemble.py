import numpy as np
import joblib
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from typing import List, Optional
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

class LGBMEnsemble:
    """
    LightGBM ensemble for molecular activity prediction, multi-fingerprint training,
    stacking, and model persistence. Data handling (folds, clustering, etc.) should be done externally.
    """
    def __init__(self, fingerprints: List[str], device: str = "cpu", seed: int = 42):
        self.fingerprints = fingerprints
        self.device = device
        self.seed = seed
        self.models = {fp: [] for fp in fingerprints}
        self.stacker = None
        self.oof_preds = {fp: None for fp in fingerprints}
        self.fold_metrics = {fp: [] for fp in fingerprints}

    @staticmethod
    def get_fixed_lgb_params(device: str, seed: int = 42) -> dict:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'seed': seed,
            'n_estimators': 300,
            'num_leaves': 224,
            'learning_rate': 0.06305278101775037,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_child_samples': 20,
        }
        if device == "gpu":
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'force_col_wise': True,
                'verbosity': 1,
                'boosting_type': 'gbdt'
            })
        return params

    def train(self, X_dict, y, folds, to_csr_fn, logger=None):
        """
        X_dict: dict of {fp: pd.Series or np.ndarray} for each fingerprint
        y: np.ndarray or pd.Series (labels)
        folds: list of (train_idx, val_idx) tuples
        to_csr_fn: function to convert fingerprint series to csr_matrix
        """
        for fp in self.fingerprints:
            oof = np.zeros(len(y))
            models = []
            fold_metrics = []
            params = self.get_fixed_lgb_params(self.device, self.seed)
            for fold, (tr, va) in enumerate(folds):
                X_tr = to_csr_fn(X_dict[fp].iloc[tr])
                y_tr = y.iloc[tr] if hasattr(y, 'iloc') else y[tr]
                X_va = to_csr_fn(X_dict[fp].iloc[va])
                y_va = y.iloc[va] if hasattr(y, 'iloc') else y[va]
                ds_tr = lgb.Dataset(X_tr, y_tr)
                ds_va = lgb.Dataset(X_va, y_va, reference=ds_tr)
                bst = lgb.train(params, ds_tr, valid_sets=[ds_va], valid_names=['valid_0'],
                                callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])
                y_pred_proba_va = bst.predict(X_va, num_iteration=bst.best_iteration)
                oof[va] = y_pred_proba_va
                models.append(bst)
                # --- Per-fold metrics ---
                y_pred_binary_va = (y_pred_proba_va >= 0.5).astype(int)
                try:
                    auc = roc_auc_score(y_va, y_pred_proba_va)
                except Exception:
                    auc = 0.0
                try:
                    auprc = average_precision_score(y_va, y_pred_proba_va)
                except Exception:
                    auprc = 0.0
                try:
                    f1 = f1_score(y_va, y_pred_binary_va, zero_division=0)
                    precision = precision_score(y_va, y_pred_binary_va, zero_division=0)
                    recall = recall_score(y_va, y_pred_binary_va, zero_division=0)
                    tn, fp_, fn, tp = confusion_matrix(y_va, y_pred_binary_va).ravel()
                except Exception:
                    f1 = precision = recall = tn = fp_ = fn = tp = 0.0
                fold_metrics.append({
                    "Fingerprint": fp,
                    "Fold": fold,
                    "AUC": auc,
                    "AUPRC": auprc,
                    "F1": f1,
                    "Precision": precision,
                    "Recall": recall,
                    "TP": int(tp),
                    "FP": int(fp_),
                    "TN": int(tn),
                    "FN": int(fn)
                })
            self.models[fp] = models
            self.oof_preds[fp] = oof
            self.fold_metrics[fp] = fold_metrics

    def fit_stacker(self, y):
        """
        Fit a logistic regression stacker on OOF predictions.
        """
        P_tr = np.column_stack([self.oof_preds[fp] for fp in self.fingerprints])
        self.stacker = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.seed)
        self.stacker.fit(P_tr, y)

    def predict(self, X_dict, to_csr_fn) -> np.ndarray:
        """
        Predict using the ensemble and stacker.
        X_dict: dict of {fp: pd.Series or np.ndarray}
        Returns: stacked prediction probabilities (np.ndarray)
        """
        preds = []
        for fp in self.fingerprints:
            X = to_csr_fn(X_dict[fp])
            fp_preds = np.mean([
                m.predict(X, num_iteration=getattr(m, 'best_iteration', -1))
                for m in self.models[fp]
            ], axis=0)
            preds.append(fp_preds)
        P = np.column_stack(preds)
        if self.stacker is not None:
            return self.stacker.predict_proba(P)[:, 1]
        else:
            return np.mean(P, axis=1)

    def save(self, path: str):
        joblib.dump({
            'models': self.models,
            'stacker': self.stacker,
            'fingerprints': self.fingerprints,
            'seed': self.seed,
            'device': self.device
        }, path)

    @classmethod
    def load(cls, path: str):
        bundle = joblib.load(path)
        obj = cls(bundle['fingerprints'], device=bundle.get('device', 'cpu'), seed=bundle.get('seed', 42))
        obj.models = bundle['models']
        obj.stacker = bundle['stacker']
        return obj
