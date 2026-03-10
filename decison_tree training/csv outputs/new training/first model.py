"""
Gold XAUUSD - LightGBM V3 Training Pipeline
============================================
Walk-forward 5-fold CV on 2003-2025, holdout test 2026+
Outputs: model, OOF predictions, feature importance, test results, drift report
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV    = os.path.join(BASE_DIR, "dataset_train_val.csv")
TEST_CSV     = os.path.join(BASE_DIR, "dataset_test.csv")
TARGET_COL   = "target_log_return"

# ── lgbm config ───────────────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators      = 30000,
    learning_rate     = 0.01,
    num_leaves        = 31,
    min_data_in_leaf  = 120,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 1,
    reg_alpha         = 1.0,
    reg_lambda        = 1.0,
    n_jobs            = -1,
    random_state      = 42,
    verbose           = -1,
)
EARLY_STOP   = 500
N_FOLDS      = 5


# ── metrics ───────────────────────────────────────────────────────────────────
def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))

def ic(y, yhat):
    return float(pd.Series(y).corr(pd.Series(yhat)))

def hit_rate(y, yhat):
    return float(np.mean(np.sign(y) == np.sign(yhat)))

def metrics(y, yhat, label=""):
    r = rmse(y, yhat)
    i = ic(y, yhat)
    h = hit_rate(y, yhat)
    print(f"  {label:<20} rmse={r:.6f}  ic={i:+.4f}  hit={h:.4f}")
    return dict(rmse=r, ic=i, hit_rate=h)


# ── drift check ───────────────────────────────────────────────────────────────
def drift_report(train_df, feat_cols):
    print("\ndrift check (feature mean: 2003-2010 vs 2024-2025)")
    early = train_df[train_df.index.year <= 2010][feat_cols]
    late  = train_df[train_df.index.year >= 2024][feat_cols]
    drift = pd.DataFrame({
        "mean_2003_2010": early.mean(),
        "mean_2024_2025": late.mean(),
        "abs_delta":      (late.mean() - early.mean()).abs(),
        "pct_delta":      ((late.mean() - early.mean()) / (early.mean().abs() + 1e-9) * 100).round(1),
    }).sort_values("abs_delta", ascending=False)
    print(drift.to_string())
    path = os.path.join(BASE_DIR, "drift_report.csv")
    drift.to_csv(path)
    print(f"saved drift_report.csv")
    return drift


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"training pipeline started  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # load data
    print("\nloading data ...")
    train_df = pd.read_csv(TRAIN_CSV, index_col=0, parse_dates=True)
    test_df  = pd.read_csv(TEST_CSV,  index_col=0, parse_dates=True)

    feat_cols = [c for c in train_df.columns if c != TARGET_COL]

    X_train = train_df[feat_cols]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[feat_cols]
    y_test  = test_df[TARGET_COL]

    print(f"train rows   : {len(train_df)}  ({train_df.index[0].date()} -> {train_df.index[-1].date()})")
    print(f"test rows    : {len(test_df)}  ({test_df.index[0].date()} -> {test_df.index[-1].date()})" if len(test_df) else "test rows    : 0")
    print(f"features     : {len(feat_cols)}  {feat_cols}")

    # ── walk-forward CV ───────────────────────────────────────────────────────
    print(f"\n5-fold walk-forward cross-validation")
    tscv        = TimeSeriesSplit(n_splits=N_FOLDS)
    oof_preds   = np.full(len(train_df), np.nan)
    fold_metrics = []
    importance_dfs = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        val_start = X_val.index[0].date()
        val_end   = X_val.index[-1].date()
        print(f"\nfold {fold}  train={len(X_tr)}  val={len(X_val)}  ({val_start} -> {val_end})")

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        best_iter = model.best_iteration_
        val_pred  = model.predict(X_val, num_iteration=best_iter)
        oof_preds[val_idx] = val_pred

        m = metrics(y_val.values, val_pred, label=f"fold {fold}")
        m["fold"]       = fold
        m["best_iter"]  = best_iter
        m["val_start"]  = str(val_start)
        m["val_end"]    = str(val_end)
        fold_metrics.append(m)

        imp = pd.DataFrame({
            "feature":   feat_cols,
            "gain":      model.booster_.feature_importance(importance_type="gain"),
            "fold":      fold,
        })
        importance_dfs.append(imp)

    # ── OOF summary ──────────────────────────────────────────────────────────
    valid_mask = ~np.isnan(oof_preds)
    print(f"\noof summary ({valid_mask.sum()} rows)")
    metrics(y_train.values[valid_mask], oof_preds[valid_mask], label="oof overall")

    # save OOF predictions
    oof_df = pd.DataFrame({
        "date":           train_df.index,
        "actual":         y_train.values,
        "oof_prediction": oof_preds,
    }).set_index("date")
    oof_path = os.path.join(BASE_DIR, "cv_predictions_oof.csv")
    oof_df.to_csv(oof_path)
    print(f"saved cv_predictions_oof.csv  ({len(oof_df)} rows)")

    # ── feature importance across folds ──────────────────────────────────────
    print("\nfeature importance (mean gain across folds)")
    imp_all = pd.concat(importance_dfs)
    imp_stable = (imp_all.groupby("feature")["gain"]
                         .agg(["mean", "std", "min", "max"])
                         .rename(columns={"mean": "mean_gain", "std": "std_gain"})
                         .sort_values("mean_gain", ascending=False))
    imp_stable["cv_stability"] = 1 - (imp_stable["std_gain"] /
                                      (imp_stable["mean_gain"].abs() + 1e-9))
    print(imp_stable.to_string())
    imp_path = os.path.join(BASE_DIR, "cv_importance_stable.csv")
    imp_stable.to_csv(imp_path)
    print(f"saved cv_importance_stable.csv")

    # ── fold metrics summary ──────────────────────────────────────────────────
    print("\nfold metrics summary")
    fm_df = pd.DataFrame(fold_metrics)
    print(fm_df[["fold", "val_start", "val_end", "rmse", "ic", "hit_rate", "best_iter"]].to_string(index=False))
    fm_df.to_csv(os.path.join(BASE_DIR, "fold_metrics.csv"), index=False)

    # ── retrain on full train set for final model ─────────────────────────────
    print("\nretraining on full train set for final model ...")
    best_iters = [m["best_iter"] for m in fold_metrics]
    final_iter = int(np.median(best_iters))
    print(f"median best iteration across folds: {final_iter}")

    final_params = {**LGBM_PARAMS, "n_estimators": final_iter}
    final_model  = lgb.LGBMRegressor(**final_params)
    final_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=-1)])

    model_path = os.path.join(BASE_DIR, "cv_best_fold_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"saved cv_best_fold_model.pkl")

    # ── 2026 test evaluation ──────────────────────────────────────────────────
    if len(test_df) > 0:
        print(f"\n2026 test evaluation  ({len(test_df)} rows)")
        test_pred = final_model.predict(X_test)
        metrics(y_test.values, test_pred, label="2026 holdout")

        oof_std  = oof_df["oof_prediction"].dropna().std()
        oof_mean = oof_df["oof_prediction"].dropna().mean()

        test_results = pd.DataFrame({
            "actual_return":    y_test.values,
            "predicted_return": test_pred,
            "pred_z_score":     (test_pred - oof_mean) / oof_std,
        }, index=test_df.index)
        results_path = os.path.join(BASE_DIR, "test_2026_results.csv")
        test_results.to_csv(results_path)
        print(f"saved test_2026_results.csv  ({len(test_results)} rows)")
    else:
        print("\nno 2026 test data available yet")

    # ── drift check ──────────────────────────────────────────────────────────
    drift_report(train_df, feat_cols)

    print(f"\npipeline complete  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()