import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import os
import torch

# --- Config ---
TRAIN_PATH = "features.csv"
TEST_PATH = "features-test.csv"

TARGETS = ["MGMT status", "1p/19q", "IDH"]
FEATURES = [f"f{i}" for i in range(512)]

OUTPUT_DIR = "models_xgboost"
N_TRIALS = 30
RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ---

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

X = df[FEATURES]
X_external = df_test[FEATURES]


def bootstrap_ci(y_true, y_pred_proba, n_bootstrap=1000):
    rng = np.random.RandomState(42)
    scores = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true.iloc[indices])) < 2:
            continue

        score = roc_auc_score(y_true.iloc[indices], y_pred_proba[indices])
        scores.append(score)

    return np.percentile(scores, [2.5, 97.5])


def objective(trial, X, y):
    try:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
        }

        model = XGBClassifier(**params, device=DEVICE)

        min_class_count = y.value_counts().min()
        n_splits = min(5, min_class_count)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")

        return scores.mean()

    except Exception as e:
        print("ERROR:", e)
        raise


# Train
for target in TARGETS:
    print(f"\n=== Optimizing for: {target} ===")

    y = df[target]

    # Split for internal validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # OPTUNA
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        catch=(Exception,),
    )

    print("Best params:", study.best_params)

    # Train best model
    model = XGBClassifier(**study.best_params, device=DEVICE)
    model.fit(X_train, y_train)

    # Internal validation
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)

    ci_low, ci_high = bootstrap_ci(y_val.reset_index(drop=True), y_val_proba)

    print(f"Validation AUC: {val_auc:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # External test (if labels exist)
    if target in df_test.columns:
        y_ext = df_test[target]

        y_ext_proba = model.predict_proba(X_external)[:, 1]
        ext_auc = roc_auc_score(y_ext, y_ext_proba)

        ci_low_ext, ci_high_ext = bootstrap_ci(
            y_ext.reset_index(drop=True), y_ext_proba
        )

        print(f"External AUC: {ext_auc:.4f}")
        print(f"External 95% CI: [{ci_low_ext:.4f}, {ci_high_ext:.4f}]")
    else:
        print("No labels in external test set → skipping metrics")

    # Save model
    path = os.path.join(OUTPUT_DIR, f"xgb_optuna_{target.replace('/', '_')}.joblib")
    joblib.dump(model, path)

print("Done.")
