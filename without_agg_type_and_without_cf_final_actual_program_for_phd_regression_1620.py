# -*- coding: utf-8 -*-
"""
Concrete Mix Design - Reverse Engineering (Regression) - 1620
=============================================================
Local version of the Google Colab notebook.

Folder layout (everything is created next to this file):

    Final-experimentation-program-1620/
    |-- Concrete_Input-COMBINED-1620.xlsx      <- input data (read AND appended to)
    |-- concrete_mix_regression.py             <- this file
    |-- requirements.txt
    `-- outputs/                               <- created automatically
        |-- graphs/
        |   |-- shape_effects/                 (A, S, T vs Slump / CF)
        |   |-- r2_scores/                     (bar charts of R2)
        |   |-- actual_vs_predicted/           (one image per variable)
        |-- models/                            (.pkl trained models)
        |-- metrics/                           (CSV of R2 / MAE / MSE)
        `-- backups/                           (Excel backup before every save)

Run:
    python concrete_mix_regression.py
"""

import sys
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # save images to files, never open pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ======================================================================
# CONFIGURATION
# ======================================================================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "Concrete_Input-COMBINED-1620.xlsx"

OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"
SHAPE_DIR = GRAPH_DIR / "shape_effects"
R2_DIR = GRAPH_DIR / "r2_scores"
AVP_DIR = GRAPH_DIR / "actual_vs_predicted"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
BACKUP_DIR = OUTPUT_DIR / "backups"

FEATURES_WITH_CF = ["A", "S", "T", "Slump", "CF", "w/c_ratio"]
FEATURES_WITHOUT_CF = ["A", "S", "T", "Slump", "w/c_ratio"]
TARGETS = ["water_content", "FA", "CA", "10", "20"]
TARGET_LABELS = {
    "water_content": "Water Content",
    "FA": "FA",
    "CA": "CA",
    "10": "10mm Aggregate",
    "20": "20mm Aggregate",
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
POLY_DEGREE = 2     # degree of the red trend curve in actual-vs-predicted plots
DPI = 200

np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = "{:.2f}".format


# ======================================================================
# HELPERS
# ======================================================================
def make_folders():
    for d in (SHAPE_DIR, R2_DIR, AVP_DIR, MODEL_DIR, METRICS_DIR, BACKUP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def safe_name(text):
    """'w/c_ratio' -> 'w_c_ratio' so it is a valid file name."""
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   [saved] {path.relative_to(BASE_DIR)}")


def r2(x):
    return round(float(x), 2)


def ask_float(prompt, default=None):
    """Ask for a number in the terminal. ENTER keeps `default` (if given)."""
    while True:
        shown = f"{prompt} [{default:.2f}]: " if default is not None else f"{prompt}: "
        raw = input(shown).strip()
        if raw == "" and default is not None:
            return round(float(default), 2)
        try:
            return r2(raw)
        except ValueError:
            print("   Please enter a valid number.")


def ask_yes_no(prompt):
    while True:
        ans = input(f"{prompt} (y/n): ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"
        print("   Please type y or n.")


# ======================================================================
# 1. LOAD DATA
# ======================================================================
def load_data():
    if not EXCEL_FILE.exists():
        sys.exit(f"ERROR: Excel file not found:\n  {EXCEL_FILE}\n"
                 f"Keep it in the same folder as this script.")

    df = pd.read_excel(EXCEL_FILE)
    df.columns = df.columns.astype(str)  # '10' and '20' must be strings

    needed = set(FEATURES_WITH_CF + TARGETS)
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"ERROR: Excel is missing columns: {sorted(missing)}\n"
                 f"Found columns: {list(df.columns)}")

    print(f"Loaded {len(df)} rows from {EXCEL_FILE.name}")
    print("\nData Preview:")
    print(df.head())
    return df


# ======================================================================
# 2. SHAPE-EFFECT GRAPHS (A, S, T)
# ======================================================================
def analyze_shape_effects(df, output_var):
    results = {}
    data = df[["A", "S", "T", output_var]].dropna()

    plt.figure(figsize=(15, 5))
    for i, p in enumerate(["A", "S", "T"]):
        X = data[[p]]
        y = data[output_var]
        m = LinearRegression().fit(X, y)
        score = m.score(X, y)
        results[p] = {"R2": score, "Coefficient": m.coef_[0], "Intercept": m.intercept_}

        plt.subplot(1, 3, i + 1)
        sns.scatterplot(x=data[p], y=y)
        order = np.argsort(X[p].values)
        plt.plot(X[p].values[order], m.predict(X)[order], color="red")
        plt.xlabel(p)
        plt.ylabel(output_var)
        plt.title(f"{output_var} vs {p}\nR²: {score:.2f}")

    save_fig(SHAPE_DIR / f"shape_effects_{safe_name(output_var)}.png")
    return results


# ======================================================================
# 3. TRAIN MODELS
# ======================================================================
def train_multi_target(df, features, label):
    """One GradientBoostingRegressor per target (same as the Colab code)."""
    data = df[features + TARGETS].dropna()
    X = data[features].values
    y = data[TARGETS]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    models = {}
    for t in TARGETS:
        m = GradientBoostingRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        m.fit(X_tr, y_tr[t])
        models[t] = m

    joblib.dump(models, MODEL_DIR / f"model_{label}.pkl")
    print(f"   [saved] {(MODEL_DIR / f'model_{label}.pkl').relative_to(BASE_DIR)}")
    return models, X_te, y_te


def train_single_target(df, features, target, label):
    data = df[features + [target]].dropna()
    X, y = data[features], data[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    m = GradientBoostingRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    m.fit(X_tr, y_tr)
    joblib.dump(m, MODEL_DIR / f"model_{label}.pkl")
    print(f"   [saved] {(MODEL_DIR / f'model_{label}.pkl').relative_to(BASE_DIR)}")
    return m, X_te, y_te


# ======================================================================
# 4. EVALUATION (terminal + CSV + graphs)
# ======================================================================
def predict_all(models, X_test):
    return np.column_stack([models[t].predict(X_test) for t in TARGETS])


def evaluate_and_report(models, X_test, y_test, label, title):
    """Prints R2/MAE/MSE in terminal, saves R2 bar chart + metrics CSV."""
    y_pred = predict_all(models, X_test)

    rows = []
    for i, col in enumerate(TARGETS):
        rows.append({
            "Target": col,
            "R2": r2_score(y_test[col], y_pred[:, i]),
            "MAE": mean_absolute_error(y_test[col], y_pred[:, i]),
            "MSE": mean_squared_error(y_test[col], y_pred[:, i]),
        })
    metrics = pd.DataFrame(rows)

    print(f"\n{'=' * 60}\nModel Performance {title}\n{'=' * 60}")
    for r in rows:
        print(f"{r['Target']:>14} | R²: {r['R2']:.4f} | MAE: {r['MAE']:.4f} | MSE: {r['MSE']:.4f}")

    csv_path = METRICS_DIR / f"metrics_{label}.csv"
    metrics.to_csv(csv_path, index=False)
    print(f"   [saved] {csv_path.relative_to(BASE_DIR)}")

    # R2 bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(metrics["Target"], metrics["R2"])
    plt.ylim(0, 1.05)
    plt.ylabel("R² Score")
    plt.title(f"R² Scores {title}")
    for i, v in enumerate(metrics["R2"]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    save_fig(R2_DIR / f"r2_scores_{label}.png")

    return y_pred


def plot_actual_vs_pred(y_true, y_pred, param_name, file_name, subtitle=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    score = r2_score(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, y_true, alpha=0.7)

    # Polynomial trend (scaled -> numerically stable)
    try:
        trend = make_pipeline(StandardScaler(),
                              PolynomialFeatures(degree=POLY_DEGREE),
                              LinearRegression())
        trend.fit(y_pred.reshape(-1, 1), y_true)
        xs = np.linspace(y_pred.min(), y_pred.max(), 200).reshape(-1, 1)
        plt.plot(xs, trend.predict(xs), color="red", linewidth=2, label="Polynomial Fit")
        plt.legend()
    except Exception as e:  # never let a trend line crash the program
        print(f"   (trend line skipped for {param_name}: {e})")

    plt.xlabel(f"Predicted {param_name}")
    plt.ylabel(f"Actual {param_name}")
    plt.title(f"{param_name} {subtitle}\nR² = {score:.2f}")
    plt.grid(True, linestyle="--", alpha=0.3)

    both = np.concatenate([y_true, y_pred])
    pad = (both.max() - both.min()) * 0.1
    plt.xlim(both.min() - pad, both.max() + pad)
    plt.ylim(both.min() - pad, both.max() + pad)

    save_fig(AVP_DIR / file_name)


# ======================================================================
# 5. FULL PIPELINE: train + evaluate + graphs
# ======================================================================
def run_full_pipeline(df, make_shape_graphs=True):
    print("\n>>> Creating shape-effect graphs (A, S, T)...") if make_shape_graphs else None
    if make_shape_graphs:
        for var in ("Slump", "CF"):
            analyze_shape_effects(df, var)

    print("\n>>> Training models WITH CF...")
    model_with_cf, X_te_cf, y_te_cf = train_multi_target(df, FEATURES_WITH_CF, "with_cf")

    print("\n>>> Training models WITHOUT CF...")
    model_without_cf, X_te_nocf, y_te_nocf = train_multi_target(df, FEATURES_WITHOUT_CF, "without_cf")

    y_pred_cf = evaluate_and_report(model_with_cf, X_te_cf, y_te_cf, "with_cf", "(With CF)")
    y_pred_nocf = evaluate_and_report(model_without_cf, X_te_nocf, y_te_nocf, "without_cf", "(Without CF)")

    print("\n>>> Actual vs Predicted graphs for output variables...")
    for i, col in enumerate(TARGETS):
        plot_actual_vs_pred(y_te_cf[col], y_pred_cf[:, i], col,
                            f"actual_vs_pred_{safe_name(col)}_with_cf.png", "(With CF)")
        plot_actual_vs_pred(y_te_nocf[col], y_pred_nocf[:, i], col,
                            f"actual_vs_pred_{safe_name(col)}_without_cf.png", "(Without CF)")

    print("\n>>> Slump prediction model...")
    slump_feats = ["A", "S", "T", "CF", "w/c_ratio"]
    m_slump, X_te, y_te = train_single_target(df, slump_feats, "Slump", "slump_predictor")
    p = m_slump.predict(X_te)
    print(f"   Slump  -> R²: {r2_score(y_te, p):.4f} | MAE: {mean_absolute_error(y_te, p):.4f} "
          f"| MSE: {mean_squared_error(y_te, p):.4f}")
    plot_actual_vs_pred(y_te, p, "Slump", "actual_vs_pred_Slump_predictor.png", "(Slump Model)")

    print("\n>>> CF prediction model...")
    cf_feats = ["A", "S", "T", "Slump", "w/c_ratio"]
    m_cf, X_te, y_te = train_single_target(df, cf_feats, "CF", "cf_predictor")
    p = m_cf.predict(X_te)
    print(f"   CF     -> R²: {r2_score(y_te, p):.4f} | MAE: {mean_absolute_error(y_te, p):.4f} "
          f"| MSE: {mean_squared_error(y_te, p):.4f}")
    plot_actual_vs_pred(y_te, p, "CF", "actual_vs_pred_CF_predictor.png", "(CF Model)")

    print(f"\nAll graphs saved in: {GRAPH_DIR}")
    return model_with_cf, model_without_cf


# ======================================================================
# 6. PREDICT MIX DESIGN (asks the demand in terminal)
# ======================================================================
def predict_mix_design(model_with_cf, model_without_cf):
    print("\n--- Enter Mix Design Demand ---")
    use_cf = ask_yes_no("Do you have Compaction Factor (CF)?")

    A = ask_float("Enter Angularity (A)")
    S = ask_float("Enter Sphericity (S)")
    T = ask_float("Enter Texture (T)")
    slump = ask_float("Enter Target Slump (for prediction)")
    wc = ask_float("Enter Water-Cement Ratio (w/c_ratio)")

    CF = np.nan
    if use_cf:
        CF = ask_float("Enter Compaction Factor (CF)")
        models, feats = model_with_cf, FEATURES_WITH_CF
    else:
        models, feats = model_without_cf, FEATURES_WITHOUT_CF

    inputs = {"A": A, "S": S, "T": T, "Slump": slump, "CF": CF, "w/c_ratio": wc}
    x = np.array([[inputs[f] for f in feats]])
    pred = np.round([models[t].predict(x)[0] for t in TARGETS], 2)
    water_content, FA, CA, agg10, agg20 = pred

    print("\n--- Predicted Mix Design ---")
    print(f"Water Content: {water_content:.2f}")
    print(f"FA:            {FA:.2f}")
    print(f"CA:            {CA:.2f}")
    print(f"10mm:          {agg10:.2f}")
    print(f"20mm:          {agg20:.2f}")

    return {"A": A, "S": S, "T": T, "target_slump": slump, "CF": CF, "w/c_ratio": wc,
            "water_content": water_content, "FA": FA, "CA": CA,
            "10": agg10, "20": agg20}


# ======================================================================
# 7. CONFIRM / EDIT AND SAVE TO EXCEL
# ======================================================================
def finalize_and_save_mix_design(df, pred):
    print("\n--- Confirm / Edit Mix Design Data ---")
    print("Press ENTER to keep the value, or type a new value to edit.\n")

    A = ask_float("Angularity (A)", pred["A"])
    S = ask_float("Sphericity (S)", pred["S"])
    T = ask_float("Texture (T)", pred["T"])
    wc = ask_float("Water-Cement Ratio (w/c_ratio)", pred["w/c_ratio"])

    if not np.isnan(pred["CF"]):
        ask_float("Predicted CF (for reference)", pred["CF"])

    water_content = ask_float("Water Content", pred["water_content"])
    FA = ask_float("Fine Aggregate (FA)", pred["FA"])
    CA = ask_float("Coarse Aggregate (CA)", pred["CA"])
    agg10 = ask_float("10mm Aggregate", pred["10"])
    agg20 = ask_float("20mm Aggregate", pred["20"])

    actual_slump = ask_float("\nEnter ACTUAL achieved slump (mandatory)")
    actual_cf = ask_float("Enter ACTUAL Compaction Factor (mandatory)")

    if "Sr. No." in df.columns and df["Sr. No."].notna().any():
        next_sr = int(df["Sr. No."].max()) + 1
    else:
        next_sr = len(df) + 1

    row = {
        "Sr. No.": next_sr, "A": A, "S": S, "T": T,
        "Slump": actual_slump, "CF": actual_cf, "w/c_ratio": wc,
        "water_content": water_content, "FA": FA, "CA": CA,
        "10": agg10, "20": agg20,
    }
    row = {col: row.get(col, np.nan) for col in df.columns}
    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Backup, then save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"{EXCEL_FILE.stem}_{stamp}.xlsx"
    shutil.copy2(EXCEL_FILE, backup)
    try:
        new_df.to_excel(EXCEL_FILE, index=False)
    except PermissionError:
        print("\n❌ Could not save: the Excel file is open in another program.")
        print("   Close it and try again. (Entry NOT saved.)")
        return df

    print(f"\n✅ Entry saved to {EXCEL_FILE.name} (Sr. No. {next_sr}).")
    print(f"   Backup of previous file: {backup.relative_to(BASE_DIR)}")
    return new_df


# ======================================================================
# 8. TERMINAL MENU
# ======================================================================
def main():
    make_folders()
    df = load_data()
    model_with_cf, model_without_cf = run_full_pipeline(df)

    while True:
        print("\n" + "=" * 60)
        print("MENU")
        print("  1) Predict a new mix design (and save it to Excel)")
        print("  2) Re-train models and regenerate all graphs")
        print("  3) Show accuracy (R², MAE, MSE) again")
        print("  4) Exit")
        choice = input("Choose an option [1-4]: ").strip()

        if choice == "1":
            pred = predict_mix_design(model_with_cf, model_without_cf)
            if ask_yes_no("\nDo you want to confirm/edit this entry and save it to Excel?"):
                new_df = finalize_and_save_mix_design(df, pred)
                if len(new_df) != len(df):
                    df = new_df
                    if ask_yes_no("Re-train models with the new data now?"):
                        model_with_cf, model_without_cf = run_full_pipeline(df, make_shape_graphs=False)
        elif choice == "2":
            df = load_data()
            model_with_cf, model_without_cf = run_full_pipeline(df)
        elif choice == "3":
            for models, feats, label, title in (
                (model_with_cf, FEATURES_WITH_CF, "with_cf", "(With CF)"),
                (model_without_cf, FEATURES_WITHOUT_CF, "without_cf", "(Without CF)"),
            ):
                data = df[feats + TARGETS].dropna()
                _, X_te, _, y_te = train_test_split(
                    data[feats].values, data[TARGETS],
                    test_size=TEST_SIZE, random_state=RANDOM_STATE)
                evaluate_and_report(models, X_te, y_te, label, title)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()
