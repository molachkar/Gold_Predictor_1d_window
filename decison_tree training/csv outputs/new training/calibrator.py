"""
Gold XAUUSD - Stage 2 Probability Calibrator Training
======================================================
Inputs : cv_predictions_oof.csv, test_2026_results.csv,
         dataset_train_val.csv, dataset_test.csv
Output : calibrator.pkl, calibrator_report.csv
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
OOF_FILE       = os.path.join(BASE_DIR, "cv_predictions_oof.csv")
TEST_FILE      = os.path.join(BASE_DIR, "test_2026_results.csv")
TV_FILE        = os.path.join(BASE_DIR, "dataset_train_val.csv")
TEST_FEAT      = os.path.join(BASE_DIR, "dataset_test.csv")
OUTPUT_MODEL   = os.path.join(BASE_DIR, "calibrator.pkl")
OUTPUT_REPORT  = os.path.join(BASE_DIR, "calibrator_report.csv")

PRED_Z_WINDOW  = 252
N_FOLDS        = 5

# features passed to calibrator — Market_State is categorical (OHE), rest numeric
NUMERIC_FEATS  = ["oof_prediction", "pred_z", "abs_pred_z", "Macro_Fast"]
CATEGORIC_FEAT = ["Market_State"]
ALL_FEATS      = NUMERIC_FEATS + CATEGORIC_FEAT


# ── data loading ──────────────────────────────────────────────────────────────
def load_predictions():
    oof = pd.read_csv(OOF_FILE, index_col=0, parse_dates=True)
    oof = oof.dropna(subset=["oof_prediction"])
    oof = oof.rename(columns={"oof_prediction": "oof_prediction",
                               "actual":         "actual_return"})

    preds = oof[["actual_return", "oof_prediction"]]

    if os.path.exists(TEST_FILE):
        test = pd.read_csv(TEST_FILE, index_col=0, parse_dates=True)
        test = test.rename(columns={"predicted_return": "oof_prediction",
                                     "actual_return":    "actual_return"})
        preds = pd.concat([preds, test[["actual_return", "oof_prediction"]]])

    preds = preds.sort_index()
    preds.index.name = "Date"
    return preds


def load_features():
    tv   = pd.read_csv(TV_FILE,   index_col=0, parse_dates=True)
    test = pd.read_csv(TEST_FEAT, index_col=0, parse_dates=True)
    feats = pd.concat([tv, test]).sort_index()
    feats.index.name = "Date"
    available = [c for c in ["Macro_Fast", "Market_State"] if c in feats.columns]
    return feats[available]


def compute_pred_z(df):
    roll         = df["oof_prediction"].shift(1).rolling(PRED_Z_WINDOW)
    df["pred_z"] = (df["oof_prediction"] - roll.mean()) / roll.std()
    df["abs_pred_z"] = df["pred_z"].abs()
    return df


def build_dataset():
    preds = load_predictions()
    feats = load_features()
    df    = preds.join(feats, how="inner")
    df    = compute_pred_z(df)

    # binary target: 1 if model predicted direction correctly
    df["is_correct"] = (
        np.sign(df["oof_prediction"]) == np.sign(df["actual_return"])
    ).astype(int)

    df = df.dropna(subset=ALL_FEATS + ["is_correct"])

    # Market_State must be string for OHE
    df["Market_State"] = df["Market_State"].astype(int).astype(str)

    print(f"dataset rows : {len(df)}  ({df.index[0].date()} -> {df.index[-1].date()})")
    print(f"hit rate     : {df['is_correct'].mean():.4f}")
    print(f"class balance: {df['is_correct'].value_counts().to_dict()}")
    return df


# ── model pipeline ────────────────────────────────────────────────────────────
def build_pipeline():
    numeric_transformer = StandardScaler()
    categoric_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num",  numeric_transformer,   NUMERIC_FEATS),
        ("cat",  categoric_transformer, CATEGORIC_FEAT),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])
    return pipeline


# ── reliability table ─────────────────────────────────────────────────────────
def reliability_table(y_true, y_prob, n_bins=10):
    bins   = np.linspace(0, 1, n_bins + 1)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)]
    bucket = pd.cut(y_prob, bins=bins, labels=labels, include_lowest=True)
    tbl = pd.DataFrame({"bucket": bucket, "actual": y_true, "prob": y_prob})
    result = (tbl.groupby("bucket", observed=True)
                 .agg(count=("actual", "size"),
                      actual_hit_rate=("actual", "mean"),
                      mean_predicted_prob=("prob", "mean"))
                 .reset_index())
    return result


# ── metrics ───────────────────────────────────────────────────────────────────
def evaluate(y_true, y_prob, label=""):
    auc    = roc_auc_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    hit    = float(np.mean(y_true == (y_prob >= 0.5).astype(int)))
    print(f"  {label:<25} auc={auc:.4f}  brier={brier:.4f}  acc={hit:.4f}")
    return dict(auc=auc, brier=brier, accuracy=hit)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("calibrator training started")

    df = build_dataset()
    X  = df[ALL_FEATS]
    y  = df["is_correct"]

    # ── walk-forward CV ───────────────────────────────────────────────────────
    print(f"\n{N_FOLDS}-fold walk-forward validation")
    tscv        = TimeSeriesSplit(n_splits=N_FOLDS)
    fold_aucs   = []
    oof_probs   = np.full(len(df), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs

        m = evaluate(y_val.values, probs, label=f"fold {fold}")
        fold_aucs.append(m["auc"])

    print(f"\n  mean cv auc : {np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")

    # ── OOF reliability ───────────────────────────────────────────────────────
    valid = ~np.isnan(oof_probs)
    print(f"\noof reliability table  ({valid.sum()} rows)")
    rel = reliability_table(y.values[valid], oof_probs[valid])
    print(rel.to_string(index=False))

    evaluate(y.values[valid], oof_probs[valid], label="oof overall")

    # ── train final calibrator on full dataset ────────────────────────────────
    print("\ntraining final calibrator on full dataset ...")
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(final_pipe, f)
    print(f"saved calibrator.pkl")

    # ── save report ───────────────────────────────────────────────────────────
    final_probs = final_pipe.predict_proba(X)[:, 1]
    report = df[["actual_return", "oof_prediction", "pred_z",
                 "abs_pred_z", "Macro_Fast", "Market_State", "is_correct"]].copy()
    report["calibrated_prob"] = final_probs
    report.to_csv(OUTPUT_REPORT)
    print(f"saved calibrator_report.csv  ({len(report)} rows)")

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\nfinal in-sample evaluation")
    evaluate(y.values, final_probs, label="full dataset")

    print(f"\nreliability table (final model, full dataset)")
    rel_final = reliability_table(y.values, final_probs)
    print(rel_final.to_string(index=False))

    print("\ncalibrator training complete")


if __name__ == "__main__":
    main()